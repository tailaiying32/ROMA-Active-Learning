
import socket
import json
import threading
import time
import torch
from typing import Optional, List

class UDPReceiver:
    """
    Background thread to receive joint angles via UDP.
    Expected format: JSON list [angle1, angle2, angle3, angle4] in degrees
    or comma-separated string "a1,a2,a3,a4".
    """
    def __init__(self, port: int = 5005, buffer_size: int = 1024):
        self.port = port
        self.buffer_size = buffer_size
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow reusing the address to avoid "Address already in use" errors on restart
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.setblocking(False)
        
        self.latest_angles: Optional[List[float]] = None
        self.running = False
        self.thread = None
        self._lock = threading.Lock()
        
    def start(self):
        """Start the listening thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        print(f"UDP Receiver started on port {self.port}")

    def stop(self):
        """Stop the listening thread and close socket."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        try:
            self.sock.close()
        except:
            pass

    def _listen_loop(self):
        # Mapping from Camera keys to our internal order (Indices 0, 1, 2, 3)
        # Internal Order: [HAA, FE, ROT, Elbow]
        KEY_MAP = {
            'shoulder_abd_add_deg': 0,       # shoulder_horizontal_abduction_r
            'shoulder_flex_ext_deg': 1,      # shoulder_flexion_r
            'shoulder_int_ext_rot_deg': 2,   # shoulder_rotation_r
            'elbow_flex_ext_deg': 3          # elbow_flexion_r
        }

        while self.running:
            try:
                data, _ = self.sock.recvfrom(self.buffer_size)
                message = data.decode('utf-8').strip()
                # Debug print (truncated)
                print(f"UDP Raw: {message[:100]}..." if len(message) > 100 else f"UDP Raw: {message}")
                
                angles_vec = [0.0, 0.0, 0.0, 0.0]
                valid_parse = False
                
                try:
                    parsed = json.loads(message)
                    
                    # Handle list wrapper: "[{...}]" -> extract first dict
                    if isinstance(parsed, list) and len(parsed) > 0:
                        parsed = parsed[0]
                        
                    # FILTER: Only process 'cam2'
                    # If it's a dict and has camera_id, enforce it.
                    if isinstance(parsed, dict) and "camera_id" in parsed:
                        cam_id = parsed.get("camera_id")
                        if cam_id != "cam2":
                            # print(f"Skipped {cam_id}") # Optional debug
                            continue

                    # Now expect dict with "joint_angles"
                    if isinstance(parsed, dict) and "joint_angles" in parsed:
                        ja = parsed["joint_angles"]
                        
                        # Check if it uses the camera keys
                        found_cam_keys = False
                        for cam_key, idx in KEY_MAP.items():
                            if cam_key in ja:
                                angles_vec[idx] = float(ja[cam_key])
                                found_cam_keys = True
                        
                        if found_cam_keys:
                            valid_parse = True
                        else:
                            # Fallback: check for standard keys (internal names)
                            # Order: [HAA, FE, ROT, Elbow]
                            JOINT_ORDER = [
                                'shoulder_abduction_r',   # HAA
                                'shoulder_flexion_r',     # FE
                                'shoulder_rotation_r',    # ROT
                                'elbow_flexion_r'         # Elbow
                            ]
                            for i, key in enumerate(JOINT_ORDER):
                                if key in ja:
                                    angles_vec[i] = float(ja[key])
                                    valid_parse = True
                    
                    # Fallback: Simple list (assuming it is cam2 if valid list)
                    elif isinstance(parsed, list) and len(parsed) == 4:
                         angles_vec = [float(x) for x in parsed]
                         valid_parse = True

                    if valid_parse:
                        with self._lock:
                            self.latest_angles = angles_vec
                            print(f"UDP Parsed (cam2): {angles_vec}")

                except json.JSONDecodeError:
                    # Fallback to CSV
                    try:
                        csv_vals = [float(x) for x in message.replace(' ', '').split(',')]
                        if len(csv_vals) == 4:
                            with self._lock:
                                self.latest_angles = csv_vals
                    except ValueError:
                        pass
                except Exception as e:
                    print(f"UDP Parse Error: {e}")

            except BlockingIOError:
                time.sleep(0.01)
            except Exception as e:
                print(f"UDP Error: {e}")
                time.sleep(0.1)

    def get_latest_angles(self) -> Optional[torch.Tensor]:
        """
        Returns the latest received angles as a Tensor in Radians.
        Returns None if no data has been received yet.
        """
        with self._lock:
            if self.latest_angles is None:
                return None
            
            # Convert degrees to radians
            return torch.deg2rad(torch.tensor(self.latest_angles, dtype=torch.float32))

# Singleton pattern for global access in Streamlit
_receiver_instance = None

def get_receiver(port: int = 5005) -> UDPReceiver:
    global _receiver_instance
    if _receiver_instance is None:
        _receiver_instance = UDPReceiver(port)
        _receiver_instance.start()
    return _receiver_instance
