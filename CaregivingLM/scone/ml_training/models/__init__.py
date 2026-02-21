from .base import BaseReachPredictor
from .dual_encoder import DualEncoderModel
from .cross_attention import CrossAttentionModel
from .physics_informed import PhysicsInformedModel
from .hierarchical import HierarchicalModel
from .variational import VariationalModel

MODEL_REGISTRY = {
    'dual_encoder': DualEncoderModel,
    'cross_attention': CrossAttentionModel,
    'physics_informed': PhysicsInformedModel,
    'hierarchical': HierarchicalModel,
    'variational': VariationalModel,
}

def get_model(model_name: str, **kwargs):
    """Factory function to get model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)