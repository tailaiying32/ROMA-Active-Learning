"""
Session state management for the Streamlit active learning UI.

Manages the application state machine and coordinates between the UI and
the active learning pipeline.
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
_this_file = Path(__file__).resolve()
_active_learning_dir = _this_file.parent.parent  # active_learning/
_roma_dir = _active_learning_dir.parent  # ROMA/

if str(_roma_dir) not in sys.path:
    sys.path.insert(0, str(_roma_dir))
if str(_active_learning_dir) not in sys.path:
    sys.path.insert(0, str(_active_learning_dir))

import streamlit as st
import torch
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from src.config import load_config, get_bounds_from_config, DEVICE
from src.latent_user_distribution import LatentUserDistribution
from src.latent_prior_generation import LatentPriorGenerator
from src.factory import build_learner
from ui.interactive_oracle import InteractiveOracle


class AppState(Enum):
    """Application state machine states."""
    SETUP = "setup"                    # Initial setup (arm params, model loading)
    READY = "ready"                    # Pipeline ready, waiting to start
    AWAITING_RESPONSE = "awaiting"     # Showing query, waiting for yes/no
    COMPLETED = "completed"            # Session completed (budget reached)
    ERROR = "error"                    # Error state


@dataclass
class QueryRecord:
    """Record of a single query and response."""
    iteration: int
    test_point: torch.Tensor
    outcome: float
    outcome_str: str


@dataclass
class SessionData:
    """Persistent data for a session."""
    # Arm configuration
    upper_arm_length: float = 0.33
    forearm_length: float = 0.28

    # Pipeline components (set during initialization)
    decoder: Any = None
    prior: Any = None
    posterior: Any = None
    oracle: InteractiveOracle = None
    learner: Any = None
    config: Dict[str, Any] = field(default_factory=dict)
    bounds: torch.Tensor = None

    # Current state
    app_state: AppState = AppState.SETUP
    current_query: Optional[torch.Tensor] = None
    current_bald_score: float = 0.0
    iteration: int = 0

    # History for display
    query_history: List[QueryRecord] = field(default_factory=list)

    # Error message if any
    error_message: str = ""


def initialize_session_state() -> None:
    """Initialize Streamlit session state if not already done."""
    if 'session' not in st.session_state:
        st.session_state.session = SessionData()


def get_session() -> SessionData:
    """Get current session data."""
    initialize_session_state()
    return st.session_state.session


def load_decoder_and_embeddings(config: dict):
    """
    Load the decoder model and embeddings.

    Args:
        config: Configuration dictionary

    Returns:
        decoder, embeddings tuple
    """
    import os

    latent_config = config.get('latent', {})
    model_path = latent_config.get('model_path', 'training_data/best_model.pt')

    # Resolve paths relative to active_learning directory
    # ui/ -> active_learning/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.isabs(model_path):
        model_path = os.path.join(base_dir, model_path)

    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Import decoder class
    from infer_params.training.model import LevelSetDecoder

    # Get model config from checkpoint
    train_config = checkpoint['config']
    model_cfg = train_config['model']
    embeddings = checkpoint['embeddings']
    num_samples = embeddings.shape[0]

    # Create decoder with correct parameters
    decoder = LevelSetDecoder(
        num_samples=num_samples,
        latent_dim=model_cfg['latent_dim'],
        hidden_dim=model_cfg.get('hidden_dim', 256),
        num_blocks=model_cfg.get('num_blocks', 3),
        num_slots=model_cfg.get('num_slots', 18),
        params_per_slot=model_cfg.get('params_per_slot', 6),
    )
    decoder.load_state_dict(checkpoint['model_state_dict'])
    decoder = decoder.to(DEVICE)
    decoder.eval()

    # Convert embeddings to tensor on device
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    embeddings = embeddings.to(DEVICE)

    print(f"Loaded decoder with {num_samples} samples, latent_dim={model_cfg['latent_dim']}")

    return decoder, embeddings


def initialize_pipeline(
    upper_arm_len: float,
    forearm_len: float,
    budget: int = 20,
    config_overrides: Dict[str, Any] = None
) -> bool:
    """
    Initialize the active learning pipeline.

    Args:
        upper_arm_len: Upper arm length in meters
        forearm_len: Forearm length in meters
        budget: Query budget
        config_overrides: Optional config overrides

    Returns:
        True if successful, False otherwise
    """
    session = get_session()

    try:
        # Load UI-specific config with correct joint names and order
        ui_config_path = Path(__file__).parent.parent / 'configs' / 'ui.yaml'
        config = load_config(str(ui_config_path))

        # Apply overrides
        if config_overrides:
            _deep_merge(config, config_overrides)

        # Set budget
        config.setdefault('stopping', {})['budget'] = budget

        # Load decoder and embeddings
        decoder, embeddings = load_decoder_and_embeddings(config)

        # Get bounds
        bounds = get_bounds_from_config(config, DEVICE)

        # Create interactive oracle
        # Default joint names in GRACE order: [HAA, FE, ROT, Elbow]
        joint_names = config.get('prior', {}).get('joint_names', [
            'shoulder_abduction_r',   # HAA
            'shoulder_flexion_r',     # FE
            'shoulder_rotation_r',    # ROT
            'elbow_flexion_r'         # Elbow
        ])
        oracle = InteractiveOracle(
            n_joints=len(joint_names),
            joint_names=joint_names
        )

        # Create prior and posterior
        prior_gen = LatentPriorGenerator(config, decoder, verbose=False)
        prior = prior_gen.get_prior(embeddings=embeddings)

        # Clone prior for posterior initialization
        posterior = LatentUserDistribution(
            latent_dim=prior.latent_dim,
            decoder=decoder,
            mean=prior.mean.clone(),
            log_std=prior.log_std.clone(),
            device=DEVICE
        )

        # Build learner with interactive oracle
        learner = build_learner(
            decoder=decoder,
            prior=prior,
            posterior=posterior,
            oracle=oracle,
            bounds=bounds,
            config=config,
            embeddings=embeddings
        )

        # Store in session
        session.decoder = decoder
        session.prior = prior
        session.posterior = posterior
        session.oracle = oracle
        session.learner = learner
        session.config = config
        session.bounds = bounds
        session.upper_arm_length = upper_arm_len
        session.forearm_length = forearm_len
        session.app_state = AppState.READY
        session.iteration = 0
        session.query_history = []
        session.error_message = ""

        return True

    except Exception as e:
        session.app_state = AppState.ERROR
        session.error_message = str(e)
        return False


def request_next_query() -> Optional[torch.Tensor]:
    """
    Request the next test query from the learner's strategy.

    Returns:
        test_point tensor if query generated, None if budget exhausted or error
    """
    session = get_session()

    if session.app_state != AppState.READY:
        return None

    # Check budget
    budget = session.config.get('stopping', {}).get('budget', 100)
    if session.iteration >= budget:
        session.app_state = AppState.COMPLETED
        return None

    try:
        # Get history for strategy
        history = session.oracle.get_history()
        past_tests = [r.test_point for r in history.get_all()]

        # Call strategy to select next test
        select_result = session.learner.strategy.select_test(
            bounds=session.bounds,
            verbose=False,
            test_history=past_tests,
            iteration=session.iteration,
            diagnostics=session.learner.diagnostics
        )

        # Parse result (can be tuple of 2 or 3 elements)
        if isinstance(select_result, tuple):
            test_point = select_result[0]
            bald_score = select_result[1] if len(select_result) > 1 else 0.0
        else:
            test_point = select_result
            bald_score = 0.0

        # Ensure test_point is on correct device
        if not isinstance(test_point, torch.Tensor):
            test_point = torch.tensor(test_point, device=DEVICE)
        test_point = test_point.to(DEVICE)

        # Store pending query
        session.current_query = test_point
        session.current_bald_score = float(bald_score)
        session.oracle.set_pending_query(test_point)
        session.app_state = AppState.AWAITING_RESPONSE

        return test_point

    except Exception as e:
        session.app_state = AppState.ERROR
        session.error_message = f"Error generating query: {str(e)}"
        return None


def submit_user_response(is_feasible: bool) -> bool:
    """
    Submit user's response (yes/no) for current query.

    Args:
        is_feasible: True for feasible/yes, False for infeasible/no

    Returns:
        True if successful
    """
    session = get_session()

    if session.app_state != AppState.AWAITING_RESPONSE:
        return False

    if session.current_query is None:
        return False

    try:
        outcome = 1.0 if is_feasible else 0.0

        # Submit to oracle (adds to history)
        session.oracle.submit_response(outcome)

        # Update posterior via VI
        history = session.oracle.get_history()

        # Calculate KL weight for this iteration
        kl_weight = session.learner._calculate_kl_weight(session.iteration)

        # Run VI update
        if isinstance(session.learner.vi, list):
            # Ensemble mode
            for vi_opt in session.learner.vi:
                vi_opt.update_posterior(
                    history,
                    kl_weight=kl_weight,
                    iteration=session.iteration
                )
        else:
            # Standard mode
            session.learner.vi.update_posterior(
                history,
                kl_weight=kl_weight,
                diagnostics=session.learner.diagnostics,
                iteration=session.iteration
            )

        # Record in history for display
        session.query_history.append(QueryRecord(
            iteration=session.iteration,
            test_point=session.current_query.clone(),
            outcome=outcome,
            outcome_str='Feasible' if is_feasible else 'Infeasible'
        ))

        # Increment iteration and reset state
        session.iteration += 1
        session.current_query = None
        session.current_bald_score = 0.0

        # Check if budget reached
        budget = session.config.get('stopping', {}).get('budget', 100)
        if session.iteration >= budget:
            session.app_state = AppState.COMPLETED
        else:
            # Auto-advance: Request next query immediately
            session.app_state = AppState.READY
            request_next_query()

        return True

    except Exception as e:
        session.app_state = AppState.ERROR
        session.error_message = f"Error processing response: {str(e)}"
        return False


def reset_session() -> None:
    """Reset session to initial state."""
    session = get_session()

    if session.oracle:
        session.oracle.reset()

    session.app_state = AppState.SETUP
    session.current_query = None
    session.current_bald_score = 0.0
    session.iteration = 0
    session.query_history = []
    session.decoder = None
    session.learner = None
    session.prior = None
    session.posterior = None
    session.error_message = ""


def get_budget() -> int:
    """Get the configured query budget."""
    session = get_session()
    return session.config.get('stopping', {}).get('budget', 20)


def get_progress() -> float:
    """Get progress as fraction (0.0 to 1.0)."""
    session = get_session()
    budget = get_budget()
    return session.iteration / budget if budget > 0 else 0.0


def _deep_merge(base: dict, override: dict) -> None:
    """Deep merge override into base (in-place)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
