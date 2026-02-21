#!/usr/bin/env python3
"""
Quick wandb setup checker to ensure logging works correctly.
"""

import wandb
import time

def test_wandb_setup():
    """Test wandb configuration and logging."""
    
    print("🔍 Testing wandb setup...")
    
    try:
        # Initialize wandb
        run = wandb.init(
            project="scone-hp-search-test",
            name=f"wandb_test_{int(time.time())}",
            tags=["test", "setup-check"],
            config={
                "test_param": 42,
                "model_type": "dual_encoder"
            }
        )
        
        # Log some test metrics
        for epoch in range(5):
            metrics = {
                "epoch": epoch,
                "train_loss": 1.0 / (epoch + 1),
                "val_loss": 0.8 / (epoch + 1),
                "test_r2": epoch * 0.1
            }
            wandb.log(metrics)
            time.sleep(0.1)  # Small delay
        
        # Test final summary
        wandb.summary["final_test_r2"] = 0.4
        wandb.summary["best_val_loss"] = 0.2
        
        print("✅ wandb logging successful!")
        print(f"📊 Check your run at: {run.url}")
        
        wandb.finish()
        return True
        
    except Exception as e:
        print(f"❌ wandb setup failed: {e}")
        print("💡 Make sure to run: wandb login")
        return False


if __name__ == "__main__":
    test_wandb_setup()