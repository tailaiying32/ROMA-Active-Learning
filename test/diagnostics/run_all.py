
import argparse
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

def main():
    parser = argparse.ArgumentParser(description="Run Latent Space Diagnostics Suite")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    parser.add_argument("--robustness", action="store_true", help="Run robustness checks (interpolation, scatter)")
    parser.add_argument("--limits", action="store_true", help="Run anatomical limits checks")
    parser.add_argument("--optimization", action="store_true", help="Run optimization dynamics check")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    # If no flags, default to help
    if not (args.all or args.robustness or args.limits or args.optimization):
        parser.print_help()
        return
        
    print("=== Starting Latent Space Diagnostics ===\n")
    
    if args.all or args.robustness:
        print(">>> Running Robustness Checks...")
        from active_learning.test.diagnostics import check_robustness
        # We need to hack sys.argv or call main manually 
        # Easier to call main but we need to pass args. 
        # Modifying main() in modules to accept args is cleaner, but for now lets just subprocess or careful import
        # We'll rely on the fact that we can call them as functions if we restructure slightly, 
        # or just invoke them via os.system for isolation?
        # Isolation is safer to avoid polluting torch state/memory.
        
        cmd = f"{sys.executable} active_learning/test/diagnostics/check_robustness.py"
        if args.model: cmd += f" --model {args.model}"
        os.system(cmd)
        print("\n")
        
    if args.all or args.limits:
        print(">>> Running Anatomical Limits Checks...")
        cmd = f"{sys.executable} active_learning/test/diagnostics/check_anatomical_limits.py"
        if args.model: cmd += f" --model {args.model}"
        os.system(cmd)
        print("\n")
        
    if args.all or args.optimization:
        print(">>> Running Optimization Dynamics Checks...")
        cmd = f"{sys.executable} active_learning/test/diagnostics/check_optimization.py"
        if args.model: cmd += f" --model {args.model}"
        os.system(cmd)
        print("\n")
        
    print("=== Diagnostics Complete ===")

if __name__ == "__main__":
    main()
