
import socket
import json
import time
import math
import random

def main():
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print(f"Sending test motion to {UDP_IP}:{UDP_PORT}...")
    print("Press Ctrl+C to stop.")
    
    start_time = time.time()
    
    try:
        while True:
            t = time.time() - start_time
            
            # ========== GRACE MODEL TEST CASES ==========
            # Coordinate System: +X Forward, +Y Left, +Z Up
            # Joint Order: [HAA, FE, ROT, Elbow]
            # Dictionary Key Order: HAA → FE → ROT → Elbow (consistent!)

            # SMOOTH MOTION (dynamic test)
            angles = {
                "shoulder_abduction_r": 45 + 30 * math.sin(t * 0.7),  # HAA
                "shoulder_flexion_r": 45 + 45 * math.sin(t),                      # FE
                "shoulder_rotation_r": 0 + 45 * math.cos(t * 0.5),                # ROT
                "elbow_flexion_r": 45 + 45 * math.sin(t * 1.2)                    # Elbow
            }

            # NEUTRAL (Arm hangs down)
            # GRACE: [HAA=0, FE=0, ROT=0, Elbow=0] → Wrist at [0, 0, -0.5]
            # angles = {
            #     "shoulder_abduction_r": 0,  # HAA
            #     "shoulder_flexion_r": 0,                # FE
            #     "shoulder_rotation_r": 0,               # ROT
            #     "elbow_flexion_r": 0                    # Elbow
            # }
            
            # SIDEWAYS RIGHT (Arm points right in frontal plane)
            # GRACE: [HAA=0, FE=90, ROT=0, Elbow=0] → Wrist at [0, -0.5, 0]
            # angles = {
            #     "shoulder_abduction_r": 0,   # HAA
            #     "shoulder_flexion_r": 90,                # FE
            #     "shoulder_rotation_r": 0,                # ROT
            #     "elbow_flexion_r": 0                     # Elbow
            # }

            # FORWARD (Arm points forward in sagittal plane)
            # GRACE: [HAA=-90, FE=90, ROT=0, Elbow=0] → Wrist at [0.5, 0, 0]
            # angles = {
            #     "shoulder_abduction_r": -90,  # HAA
            #     "shoulder_flexion_r": 90,                 # FE
            #     "shoulder_rotation_r": 0,                 # ROT
            #     "elbow_flexion_r": 0                      # Elbow
            # }
            
            # UP (Arm points upward)
            # GRACE: [HAA=0, FE=180, ROT=0, Elbow=0] → Wrist at [0, 0, 0.5]
            # angles = {
            #     "shoulder_abduction_r": 0,   # HAA
            #     "shoulder_flexion_r": 180,               # FE
            #     "shoulder_rotation_r": 0,                # ROT
            #     "elbow_flexion_r": 0                     # Elbow
            # }

            # LEFT (Arm points left)
            # GRACE: [HAA=0, FE=-90, ROT=0, Elbow=0] → Wrist at [0, 0.5, 0]
            # angles = {
            #     "shoulder_abduction_r": 0,   # HAA
            #     "shoulder_flexion_r": -90,               # FE
            #     "shoulder_rotation_r": 0,                # ROT
            #     "elbow_flexion_r": 0                     # Elbow
            # }

            # BACKWARD (Arm points backward)
            # GRACE: [HAA=90, FE=90, ROT=0, Elbow=0] → Wrist at [-0.5, 0, 0]
            # angles = {
            #     "shoulder_abduction_r": 90,  # HAA
            #     "shoulder_flexion_r": 90,                # FE
            #     "shoulder_rotation_r": 0,                # ROT
            #     "elbow_flexion_r": 0                     # Elbow
            # }

            # BENT ELBOW (Arm down with elbow bent)
            # GRACE: [HAA=-45, FE=45, ROT=0, Elbow=90]
            # angles = {
            #     "shoulder_abduction_r": -45,  # HAA
            #     "shoulder_flexion_r": 45,                # FE
            #     "shoulder_rotation_r": 0,                # ROT
            #     "elbow_flexion_r": 90                    # Elbow
            # }
            
            
            
            # Construct message matching the camera system format
            message = {
                "timestamp": time.time(),
                "camera_id": "cam2",
                "frame_id": int(t * 10),
                "shoulder_uv": [320, 240],
                "joint_angles": angles,
                "shoulder_3d_metric": {"x": 0.5, "y": 0.2, "z": 0.1}
            }
            
            json_str = json.dumps(message)
            sock.sendto(json_str.encode('utf-8'), (UDP_IP, UDP_PORT))
            
            print(f"\rSent: {json_str[:60]}...", end="")
            time.sleep(0.1) # 10 Hz
            
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
