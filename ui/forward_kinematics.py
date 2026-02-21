"""Configurable forward kinematics for 4-DoF arm (Intrinsic Euler X-Y-X).

Coordinate System (FLU - Forward-Left-Up):
    +X: Forward (Anterior/Chest)
    +Y: Left (Lateral)
    +Z: Upward (Superior)
    Origin: Shoulder Joint Center

Base orientation R_base = Ry(90deg) @ Rx(-90deg) aligns the arm so that
at zero pose (all angles = 0) the arm hangs straight down (-Z).

Kinematic chain (GRACE model):
    R_base @ Rx(HAA) @ Ry(FE) @ Rx(ROT) @ Tx(upper) @ Ry(Elbow) @ Tx(forearm)

Joint definitions:
    HAA (Horizontal Abduction/Adduction): Rotation about X axis (Plane Selector)
    FE (Flexion/Extension): Rotation about Y axis (Elevation)
    ROT (Rotation): Rotation about X axis (Axial Rotation)
    Elbow: Rotation about Y axis (Hinge)

Bones extend along the local +X axis.

Reference: See forward_kinematics.md for detailed specification.
"""

import torch
import numpy as np
from typing import Tuple, Union

# Default arm segment lengths (meters)
DEFAULT_UPPER_ARM_LEN = 0.30
DEFAULT_FOREARM_LEN = 0.20

# Base orientation matrix: R_base = Ry(90deg) @ Rx(-90deg)
# This constant matrix ensures:
# - At zero pose, arm points down (-Z)
# - Local X axis (bone axis) points down
# - Local Y axis (hinge) points backward (-X)
R_BASE_MATRIX = torch.tensor([
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0,  1.0, 0.0],
    [-1.0, 0.0,  0.0, 0.0],
    [0.0,  0.0,  0.0, 1.0],
], dtype=torch.float32)


def build_rotation_matrix(
    angle_batch: torch.Tensor,
    axis_vector: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Build batch of 4x4 homogeneous rotation matrices using Rodrigues' formula.

    Args:
        angle_batch: 1D tensor of angles in radians, shape [B]
        axis_vector: Unit axis vector, shape [3] (e.g., [1, 0, 0])
        device: Device for tensor creation

    Returns:
        Batch of 4x4 rotation matrices, shape [B, 4, 4]
    """
    batch_size = angle_batch.shape[0]

    axis = axis_vector.to(device)

    cos_a = torch.cos(angle_batch)
    sin_a = torch.sin(angle_batch)
    versin_a = 1.0 - cos_a

    x, y, z = axis[0], axis[1], axis[2]
    rot_mat_3x3 = torch.zeros(batch_size, 3, 3, device=device)

    rot_mat_3x3[:, 0, 0] = cos_a + x * x * versin_a
    rot_mat_3x3[:, 0, 1] = x * y * versin_a - z * sin_a
    rot_mat_3x3[:, 0, 2] = x * z * versin_a + y * sin_a
    rot_mat_3x3[:, 1, 0] = y * x * versin_a + z * sin_a
    rot_mat_3x3[:, 1, 1] = cos_a + y * y * versin_a
    rot_mat_3x3[:, 1, 2] = y * z * versin_a - x * sin_a
    rot_mat_3x3[:, 2, 0] = z * x * versin_a - y * sin_a
    rot_mat_3x3[:, 2, 1] = z * y * versin_a + x * sin_a
    rot_mat_3x3[:, 2, 2] = cos_a + z * z * versin_a

    T_4x4 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    T_4x4[:, :3, :3] = rot_mat_3x3

    return T_4x4


def build_translation_matrix(
    device: torch.device,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0
) -> torch.Tensor:
    """Build a single 4x4 homogeneous translation matrix.

    Args:
        device: Device for tensor creation
        dx, dy, dz: Translation amounts

    Returns:
        4x4 translation matrix with shape [1, 4, 4] for broadcasting
    """
    T = torch.eye(4, device=device)
    T[0, 3] = dx
    T[1, 3] = dy
    T[2, 3] = dz
    return T.unsqueeze(0)


def compute_joint_positions(
    joint_angles: torch.Tensor,
    upper_arm_len: float = DEFAULT_UPPER_ARM_LEN,
    forearm_len: float = DEFAULT_FOREARM_LEN
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 3D positions of shoulder, elbow, and wrist from joint angles.

    Uses the GRACE kinematic model with Intrinsic X-Y-X Euler sequence.

    Args:
        joint_angles: Shape (4,) tensor in radians
            [HAA, FE, ROT, Elbow] where:
            - HAA: Shoulder Horizontal Abduction/Adduction (Rx, Plane Selector)
            - FE: Shoulder Flexion/Extension (Ry, Elevation)
            - ROT: Shoulder Rotation (Rx, Axial Rotation)
            - Elbow: Elbow Flexion (Ry, Hinge)
        upper_arm_len: Upper arm segment length (meters)
        forearm_len: Forearm segment length (meters)

    Returns:
        shoulder_pos: (3,) numpy array, always at origin [0, 0, 0]
        elbow_pos: (3,) numpy array
        wrist_pos: (3,) numpy array (end effector)
    """
    device = joint_angles.device
    batch = joint_angles.unsqueeze(0)  # Add batch dimension: [1, 4]

    # Axis definitions
    axis_x = torch.tensor([1., 0., 0.], device=device)
    axis_y = torch.tensor([0., 1., 0.], device=device)

    # Shoulder is always at origin
    shoulder_pos = np.array([0.0, 0.0, 0.0])

    # Base orientation matrix
    R_base = R_BASE_MATRIX.unsqueeze(0).to(device)  # [1, 4, 4]

    # Build shoulder rotation transforms (Intrinsic X-Y-X sequence)
    # Chain: R_base @ Rx(HAA) @ Ry(FE) @ Rx(ROT)
    T_haa = build_rotation_matrix(batch[:, 0], axis_x, device)   # HAA rotates about X
    T_fe = build_rotation_matrix(batch[:, 1], axis_y, device)    # FE rotates about Y
    T_rot = build_rotation_matrix(batch[:, 2], axis_x, device)   # ROT rotates about X

    # Translation to elbow (bone extends along local +X axis)
    T_upper_arm = build_translation_matrix(device, dx=upper_arm_len)

    # Elbow position in world frame
    # Chain: R_base @ Rx(HAA) @ Ry(FE) @ Rx(ROT) @ Tx(upper_arm)
    T_world_to_elbow = R_base @ T_haa @ T_fe @ T_rot @ T_upper_arm
    elbow_pos = T_world_to_elbow[0, :3, 3].cpu().numpy()

    # Elbow rotation (about local Y axis - hinge joint)
    T_elbow = build_rotation_matrix(batch[:, 3], axis_y, device)

    # Translation to wrist (forearm extends along local +X axis)
    T_forearm = build_translation_matrix(device, dx=forearm_len)

    # Wrist position in world frame
    # Full chain: R_base @ Rx(HAA) @ Ry(FE) @ Rx(ROT) @ Tx(upper) @ Ry(Elbow) @ Tx(forearm)
    T_world_to_wrist = T_world_to_elbow @ T_elbow @ T_forearm
    wrist_pos = T_world_to_wrist[0, :3, 3].cpu().numpy()

    return shoulder_pos, elbow_pos, wrist_pos


def forward_kinematics(
    joint_angles_batch: torch.Tensor,
    upper_arm_len: float = DEFAULT_UPPER_ARM_LEN,
    forearm_len: float = DEFAULT_FOREARM_LEN
) -> torch.Tensor:
    """
    Compute end-effector (wrist) positions for a batch of joint angles.

    Uses the GRACE kinematic model with Intrinsic X-Y-X Euler sequence.

    Args:
        joint_angles_batch: Shape [B, 4] tensor in radians
            Joint order: [HAA, FE, ROT, Elbow]
        upper_arm_len: Upper arm segment length (meters)
        forearm_len: Forearm segment length (meters)

    Returns:
        End-effector positions, shape [B, 3]
    """
    batch_size = joint_angles_batch.shape[0]
    device = joint_angles_batch.device

    # Axis definitions
    axis_x = torch.tensor([1., 0., 0.], device=device)
    axis_y = torch.tensor([0., 1., 0.], device=device)

    # Link translations (bones extend along +X)
    T_upper_arm = build_translation_matrix(device, dx=upper_arm_len)
    T_forearm = build_translation_matrix(device, dx=forearm_len)

    # Base orientation matrix
    R_base = R_BASE_MATRIX.to(device).unsqueeze(0)  # [1, 4, 4] for broadcasting

    # Build transformation chain (Intrinsic X-Y-X)
    # Chain: R_base @ Rx(HAA) @ Ry(FE) @ Rx(ROT) @ Tx(upper) @ Ry(Elbow) @ Tx(forearm)
    T_haa = build_rotation_matrix(joint_angles_batch[:, 0], axis_x, device)   # HAA (Rx)
    T_fe = build_rotation_matrix(joint_angles_batch[:, 1], axis_y, device)    # FE (Ry)
    T_rot = build_rotation_matrix(joint_angles_batch[:, 2], axis_x, device)   # ROT (Rx)
    T_elbow = build_rotation_matrix(joint_angles_batch[:, 3], axis_y, device) # Elbow (Ry)

    # Combine all transformations
    T_final = R_base @ T_haa @ T_fe @ T_rot @ T_upper_arm @ T_elbow @ T_forearm

    # Extract end-effector positions
    end_effector_pos = T_final[:, :3, 3]

    return end_effector_pos


# Verification tests based on GRACE model specification
if __name__ == "__main__":
    print("=" * 70)
    print("Forward Kinematics Verification - GRACE Model (Intrinsic X-Y-X)")
    print("=" * 70)
    print(f"Upper Arm: {DEFAULT_UPPER_ARM_LEN}m, Forearm: {DEFAULT_FOREARM_LEN}m")
    print(f"Total Arm Length: {DEFAULT_UPPER_ARM_LEN + DEFAULT_FOREARM_LEN}m")
    print()

    total_len = DEFAULT_UPPER_ARM_LEN + DEFAULT_FOREARM_LEN

    # Verification cases from specification (forward_kinematics.md)
    # Each test includes expected behavior based on GRACE model
    test_cases = [
        # (angles_deg, description, expected_wrist_direction)
        ([0, 0, 0, 0], "Neutral (Zero Pose)", "Down (-Z)", [0, 0, -total_len]),
        ([0, 90, 0, 0], "Side Reach (Right)", "Right (-Y)", [0, -total_len, 0]),
        ([-90, 90, 0, 0], "Forward Reach", "Forward (+X)", [total_len, 0, 0]),
        ([90, 90, 0, 0], "Backward Reach", "Backward (-X)", [-total_len, 0, 0]),
        ([0, 45, 0, 0], "45deg Side Lift", "Down-Right", None),  # No exact expected
        ([0, 0, 0, 90], "Elbow Bent 90deg", "Down", None),  # Forearm changes only
    ]

    print("Test Results:")
    print("-" * 70)
    all_passed = True

    for angles_deg, description, direction, expected_pos in test_cases:
        angles_rad = torch.deg2rad(torch.tensor(angles_deg, dtype=torch.float32))
        shoulder, elbow, wrist = compute_joint_positions(angles_rad)

        # Check against expected position if provided
        passed = True
        if expected_pos is not None:
            expected = np.array(expected_pos, dtype=np.float32)
            if not np.allclose(wrist, expected, atol=1e-5):
                passed = False
                all_passed = False

        status = "✓" if passed else "✗"

        print(f"{status} {description:25s} -> {direction}")
        print(f"   Angles (deg): [HAA={angles_deg[0]:>5.0f}, FE={angles_deg[1]:>5.0f}, "
              f"ROT={angles_deg[2]:>5.0f}, Elbow={angles_deg[3]:>5.0f}]")
        print(f"   Shoulder: [{shoulder[0]:>7.3f}, {shoulder[1]:>7.3f}, {shoulder[2]:>7.3f}]")
        print(f"   Elbow:    [{elbow[0]:>7.3f}, {elbow[1]:>7.3f}, {elbow[2]:>7.3f}]")
        print(f"   Wrist:    [{wrist[0]:>7.3f}, {wrist[1]:>7.3f}, {wrist[2]:>7.3f}]")

        if expected_pos is not None:
            print(f"   Expected: [{expected_pos[0]:>7.3f}, {expected_pos[1]:>7.3f}, {expected_pos[2]:>7.3f}]")
            if not passed:
                diff = wrist - expected
                print(f"   ERROR: Difference = [{diff[0]:>7.3f}, {diff[1]:>7.3f}, {diff[2]:>7.3f}]")

        print()

    print("=" * 70)
    if all_passed:
        print("✓ All verification tests PASSED")
    else:
        print("✗ Some tests FAILED - implementation may be incorrect")
    print("=" * 70)
