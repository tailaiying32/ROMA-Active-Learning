# Kinematic Specification: "Grace" Upper Body Model

**Context:** This document specifies the non-standard "Plane of Elevation" kinematic model used in the `grace` project. It provides sufficient detail to replicate the Forward Kinematics (FK) in any external system (C++, Unity, PyTorch, etc.).

---

## 1. Global Coordinate System
The system utilizes a **Right-Handed**, **FLU** (Forward-Left-Up) coordinate system.

*   **+X:** Forward (Anterior)
*   **+Y:** Left (Lateral)
*   **+Z:** Up (Superior)
*   **Origin:** Shoulder Joint Center

---

## 2. The Kinematic Chain (Shoulder)

The shoulder is modeled as a 3-DOF spherical joint using an **Intrinsic Euler X-Y-X** sequence.

### 2.1 Joint Variable Definitions
The naming convention in this system is **non-standard**. The variables are defined functionally rather than anatomically.

| Parameter | Symbol | Axis of Rotation (Local) | Function | Zero State Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **HAA** | $\theta_1$ | **X** (Bone Axis) | **Plane Selector** | $\theta_1=0$ aligns the "Lift" to the **Frontal Plane** (Rightward). |
| **Flexion (FE)** | $\theta_2$ | **Y** (Hinge Axis) | **Elevation** | $\theta_2=0$ is arm down. Increasing $\theta_2$ lifts the arm. |
| **Rotation (ROT)**| $\theta_3$ | **X** (Bone Axis) | **Axial Rotation** | Internal/External rotation of the humerus. |

### 2.2 The "Zero Pose" Configuration
At $(\theta_1=0, \theta_2=0, \theta_3=0)$:
*   **Arm Vector:** Points **Down** (Global $-Z$).
*   **Hinge Alignment:** The axis of elevation (Local Y) points **Backward** (Global $-X$).
    *   *Geometric Note:* This alignment is strictly required to satisfy the "Flexion is Frontal" condition. Rotating around a **Backward** axis lifts the arm **Sideways** (to the Right). If the axis pointed Rightward, the arm would lift Forward.
*   **Result:** A pure "Flexion" input will lift the arm to the **Right** (Sideways) in the Frontal Plane.

---

## 3. Mathematical Formulation (Replication Guide)

To replicate the kinematics, you must construct the rotation matrix using the following chain.

### 3.1 Rotation Primitives
Standard elemental rotation matrices (Right-Hand Rule):

$$ R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & c_\theta & -s_\theta \\ 0 & s_\theta & c_\theta \end{bmatrix}, \quad R_y(\theta) = \begin{bmatrix} c_\theta & 0 & s_\theta \\ 0 & 1 & 0 \\ -s_\theta & 0 & c_\theta \end{bmatrix} $$

### 3.2 The Chain of Operations
The rotation of the Upper Arm ($R_{arm}$) relative to the Global Frame is calculated as:

$$ R_{arm} = R_{base} \times R_{x}(\theta_{haa}) \times R_{y}(\theta_{fe}) \times R_{x}(\theta_{rot}) $$

### 3.3 The Base Orientation ($R_{base}$)
This is the critical constant matrix that aligns the math with the specific definitions (HAA=0 is Rightward).

We construct $R_{base}$ by applying two operations to the Identity matrix:
1.  **Align to Down:** Rotate around Y by +90° (Points X-axis to -Z).
2.  **Align Hinge to Back:** Rotate around X (the new vertical axis) by -90°.

$$ R_{base} = R_y(90^{\circ}) \times R_x(-90^{\circ}) $$

*Note: Depending on your exact "Identity" definition, you simply need to ensure that before any angles are applied, the Local X axis is **Down** (-Z) and the Local Y axis is **Backward** (-X).*

### 3.4 Elbow Kinematics
The elbow is a 1-DOF hinge attached to the Upper Arm frame.
*   **Rotation Axis:** Local Y (of the Upper Arm).
*   **Operation:** $R_{forearm} = R_{arm} \times R_y(\theta_{elbow})$

---

## 4. Reference Implementation (Python)

Use this code snippet to verify your implementation.

```python
import numpy as np

def fk_grace_model(haa, fe, rot, elbow, l_upper=1.0, l_lower=1.0):
    """
    Computes Forward Kinematics for the Grace Model.
    Args:
        haa, fe, rot, elbow: Angles in degrees.
        l_upper, l_lower: Bone lengths.
    Returns:
        elbow_pos, hand_pos (Global Coordinates [x,y,z])
    """
    # 1. Helper Matrices (Standard)
    def Rx(deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    def Ry(deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])

    # 2. Base Orientation Construction
    # Ry(90): Moves X(Front) -> Z(Down)
    # Rx(-90): Twists so Y(Left) -> X(Back)
    R_base = Ry(90) @ Rx(-90)

    # 3. The Chain (Intrinsic XYX)
    R_arm = R_base @ Rx(haa) @ Ry(fe) @ Rx(rot)

    # 4. Positions
    # Bone vector is Local X (First column of rotation matrix)
    vec_upper = R_arm[:, 0] * l_upper
    elbow_pos = vec_upper 
    
    # 5. Elbow (Hinge around Local Y)
    R_forearm = R_arm @ Ry(elbow)
    vec_lower = R_forearm[:, 0] * l_lower
    hand_pos = elbow_pos + vec_lower

    return elbow_pos, hand_pos
```

## 5. Verification Table

Use these values to test your implementation.

| Pose Description | Input (HAA, FE, ROT) | Expected Behavior |
| :--- | :--- | :--- |
| **Neutral** | (0, 0, 0) | Arm hangs **Down** (-Z). |
| **Side Reach (Right)** | (0, 90, 0) | Arm points **Right** (-Y). (Coronal Plane) |
| **Forward Reach** | (-90, 90, 0) | Arm points **Forward** (+X). (Sagittal Plane) |
| **Backward Reach** | (90, 90, 0) | Arm points **Backward** (-X). |
