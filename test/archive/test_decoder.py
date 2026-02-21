import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
from train.models.autodecoder import SIRSAutoDecoder

# 1. Load checkpoint
checkpoint = torch.load('models/pic_model_epoch_100.pth', map_location='cpu')

# print(f"Checkpoint keys: {checkpoint.keys()}")
# print(f"Model state dict keys: {checkpoint['model_state_dict'].keys()}")
# print(f"Config: {checkpoint['config'].keys()}")

# 2. Recreate model with correct config
model = SIRSAutoDecoder(
    n_samples=checkpoint['n_samples'],
    in_dim=4,
    latent_dim=checkpoint['latent_dim'],
    decoder_hidden_dim=checkpoint['config']['decoder_hidden_dim'],
    decoder_num_layers=checkpoint['config']['decoder_num_layers'],
)


# Prepare sample indices and query points for testing (no dataset needed)
sample_indices = torch.tensor([0])  # Single sample, shape (1,)
query_points_raw = torch.randn(1, 100, 4)  # 100 random 4D points, shape (1, 100, 4)

# --- Test and verify the three key decoder operations ---
with torch.no_grad():
    # 1. Get latent code for a sample
    print("==== get_latent ====")
    z = model.get_latent(sample_indices)  # shape: (B, latent_dim)
    print("z shape:", z.shape)

    print("\n==== decode_transform ====")
    # 2. Transform head: z → (center, halfwidths)
    center, halfwidths = model.decode_transform(z)
    print("center:", center)
    print("halfwidths:", halfwidths)

    print("\n=== raw_to_canonical ===")
    # 3. Canonicalize raw points
    q_canonical = model.raw_to_canonical(query_points_raw, center, halfwidths)
    print("q_canonical shape:", q_canonical.shape)

    print("\n==== decoder ====")
    # 4. Decoder query
    logit = model.decoder(q_canonical, z)
    print("logit shape:", logit.shape)

    print("\n==== Full forward pass ====")
    # Optionally, run the full forward for comparison
    output = model(sample_indices, query_points_raw)
    print("output['sdf_pred'] shape:", output['sdf_pred'].shape)