from active_learning.src.svgd.svgd_vi import SVGDVariationalInference, _reset_csvs
from active_learning.src.svgd.sliced_svgd_optimizer import SlicedSVGD
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.svgd.particle_user_distribution import ParticleUserDistribution


class SlicedSVGDVariationalInference(SVGDVariationalInference):
    """
    Sliced SVGD Inference.

    Inherits all log_likelihood, log_prior, likelihood, regularizer,
    update_posterior, and diagnostic logging from SVGDVariationalInference.

    Only overrides __init__ to:
      - Read from config['sliced_svgd'] instead of config['svgd']
      - Create SlicedSVGD optimizer instead of SVGD
    """

    def __init__(
        self,
        decoder,
        prior: LatentUserDistribution,
        posterior: ParticleUserDistribution,
        config: dict = None
    ):
        # Do NOT call super().__init__() — it reads config['svgd']
        # and creates the vanilla SVGD optimizer.
        self.decoder = decoder
        self.prior = prior
        self.posterior = posterior
        self.config = config or {}

        # Read from 'sliced_svgd' config section (independent of 'svgd')
        sliced_config = self.config.get('sliced_svgd', {})
        self.max_iters = sliced_config.get('max_iters', 100)
        self.n_particles = self.posterior.n_particles
        self.kernel_width = sliced_config.get('kernel_width', None)

        # Step size decay configuration
        step_decay_cfg = sliced_config.get('step_decay', {})
        self.step_decay_enabled = step_decay_cfg.get('enabled', False)
        self.step_decay_schedule = step_decay_cfg.get('schedule', 'power')  # 'linear', 'cosine', 'exponential', 'power'
        # Support both new (start_lr in step_decay) and legacy (step_size at top level) config
        self.step_decay_start_lr = step_decay_cfg.get('start_lr', sliced_config.get('step_size', 0.1))
        self.step_decay_end_lr = step_decay_cfg.get('end_lr', self.step_decay_start_lr * 0.25)
        self.step_decay_duration = step_decay_cfg.get('duration', None)  # None = use max_iters
        self.step_decay_power = step_decay_cfg.get('power', 0.5)  # Legacy: lr / n_data^power
        self.lr = self.step_decay_start_lr  # Base LR for the optimizer

        # Gradient clipping (shared with vi config)
        vi_config = self.config.get('vi', {})
        self.grad_clip = vi_config.get('grad_clip', 1.0)

        # Likelihood temperature (shared with bald config)
        self.tau = self.config.get('bald', {}).get('tau', 1.0)
        self.tau_schedule = self.config.get('bald', {}).get('tau_schedule', None)

        # Sliced SVGD does not need repulsive_scaling (1D kernel is fine)
        self.repulsive_scaling = 0.0

        # Create SlicedSVGD optimizer
        n_slices = sliced_config.get('n_slices', 20)
        kernel_type = sliced_config.get('kernel_type', 'rbf')
        self.optimizer = SlicedSVGD(
            n_slices=n_slices,
            kernel_width=self.kernel_width,
            kernel_type=kernel_type
        )

        # Prior weight
        self.kl_weight = sliced_config.get('prior_weight', 1.0)

        # Reset diagnostic CSVs at init (new run)
        _reset_csvs()
        self._first_particles = None
