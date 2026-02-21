# SCONE Reach Distance Prediction

Machine learning pipeline for predicting reach distances from muscle parameters and target poses in SCONE biomechanical simulations.

## Architecture

The codebase supports multiple model architectures with easy swapping via Hydra configs:

1. **Dual Encoder** (baseline): Separate encoders for muscles and pose, concatenated for prediction
2. **Cross Attention**: Attention mechanism between muscle and pose features  
3. **Physics Informed**: Base physics heuristic + muscle correction residual
4. **Hierarchical**: Anatomical muscle grouping by function
5. **Variational**: VAE-based muscle parameter manifold learning

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the baseline dual-encoder model
python train.py

# Train with different config
python train.py model=dual_encoder data.batch_size=64 training.learning_rate=0.001

# Train different architecture (when implemented)
python train.py model=cross_attention
```

## Configuration

Configuration is managed via Hydra. Key config files:

- `config/config.yaml`: Main config with defaults
- `config/model/`: Model architecture configs  
- `config/data/`: Dataset and preprocessing configs
- `config/training/`: Training hyperparameters
- `config/logging/`: Weights & Biases logging setup

## Data Format

The pipeline expects data in `compiled_results/` with:
- `{model}_results.txt`: Target poses and distances (x y z distance format)
- `{model}_muscle_params.json`: Muscle parameter scale factors

## Monitoring

Training progress is logged to Weights & Biases. Key metrics:
- Loss (MSE/MAE/Huber)
- R-squared coefficient  
- RMSE
- Learning rate
- Epoch timing

## Model Features

- **Modular Architecture**: Easy to swap models via config
- **CUDA Support**: Automatic GPU detection and usage
- **Early Stopping**: Configurable patience and criteria
- **Checkpointing**: Best and periodic model saving
- **Data Scaling**: Automatic feature standardization
- **Multiple Loss Functions**: MSE, MAE, Huber loss options