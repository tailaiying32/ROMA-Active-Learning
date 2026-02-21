"""
Arm visualization using Plotly for 3D and 2D views.

Provides interactive 3D skeleton visualization and 2D orthographic projections
of arm configurations from joint angles.
"""

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, List, Optional

from .forward_kinematics import (
    compute_joint_positions,
    DEFAULT_UPPER_ARM_LEN,
    DEFAULT_FOREARM_LEN,
)


class ArmVisualizer:
    """
    Visualizes arm configurations using forward kinematics and Plotly.

    Produces:
    - 3D interactive figure with skeleton lines and joint markers
    - 2D orthographic projections (front, side, top views)
    """

    # Joint display names — order matches data: [HAA, FE, ROT, Elbow]
    DEFAULT_JOINT_NAMES = [
        'Shoulder Horizontal Abduction',
        'Shoulder Flexion',
        'Shoulder Rotation',
        'Elbow Flexion'
    ]

    def __init__(
        self,
        upper_arm_length: float = DEFAULT_UPPER_ARM_LEN,
        forearm_length: float = DEFAULT_FOREARM_LEN
    ):
        """
        Initialize the visualizer with arm segment lengths.

        Args:
            upper_arm_length: Length of upper arm segment (meters)
            forearm_length: Length of forearm segment (meters)
        """
        self.upper_arm_len = upper_arm_length
        self.forearm_len = forearm_length
        self.total_arm_len = upper_arm_length + forearm_length

    def get_joint_positions(
        self,
        joint_angles: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 3D positions of shoulder, elbow, and wrist.

        Args:
            joint_angles: Shape (4,) tensor in radians

        Returns:
            shoulder_pos, elbow_pos, wrist_pos: Each (3,) numpy arrays
        """
        return compute_joint_positions(
            joint_angles,
            upper_arm_len=self.upper_arm_len,
            forearm_len=self.forearm_len
        )

    def create_3d_figure(
        self,
        joint_angles: torch.Tensor,
        user_angles: Optional[torch.Tensor] = None,
        title: str = "Arm Configuration",
        show_reference: bool = True,
        height: int = 500
    ) -> go.Figure:
        """
        Create interactive 3D Plotly figure of the arm skeleton.

        Args:
            joint_angles: Shape (4,) tensor in radians (The Target/Ghost pose)
            user_angles: Optional Shape (4,) tensor in radians (The Real-time User pose)
            title: Figure title
            show_reference: Whether to show a faint reference pose (zero pose)
            height: Figure height in pixels

        Returns:
            Plotly Figure object
        """
        shoulder, elbow, wrist = self.get_joint_positions(joint_angles)

        fig = go.Figure()

        # Reference pose (zero pose) - Faint Gray
        if show_reference:
            ref_shoulder, ref_elbow, ref_wrist = self.get_joint_positions(
                torch.zeros(4)
            )
            # Upper arm reference
            fig.add_trace(go.Scatter3d(
                x=[ref_shoulder[0], ref_elbow[0]],
                y=[ref_shoulder[1], ref_elbow[1]],
                z=[ref_shoulder[2], ref_elbow[2]],
                mode='lines',
                line=dict(color='lightgray', width=3, dash='dot'),
                name='Zero Pose',
                showlegend=True,
                legendgroup='zero'
            ))
            # Forearm reference
            fig.add_trace(go.Scatter3d(
                x=[ref_elbow[0], ref_wrist[0]],
                y=[ref_elbow[1], ref_wrist[1]],
                z=[ref_elbow[2], ref_wrist[2]],
                mode='lines',
                line=dict(color='lightgray', width=3, dash='dot'),
                showlegend=False,
                legendgroup='zero'
            ))

        # --- TARGET ARM (The Query) ---
        # If user_angles is present, make target 'ghostly' (transparent blue)
        # Otherwise, make it solid blue/green (standard)
        
        if user_angles is not None:
            target_opacity = 0.4
            target_name = "Target (Goal)"
            upper_color = f'rgba(31, 119, 180, {target_opacity})' # Blue transparent
            fore_color = f'rgba(44, 160, 44, {target_opacity})'  # Green transparent
            joint_opacity = 0.4
        else:
            target_name = "Target Arm"
            upper_color = '#1f77b4' # Solid Blue
            fore_color = '#2ca02c'  # Solid Green
            joint_opacity = 1.0

        # Upper arm segment
        fig.add_trace(go.Scatter3d(
            x=[shoulder[0], elbow[0]],
            y=[shoulder[1], elbow[1]],
            z=[shoulder[2], elbow[2]],
            mode='lines',
            line=dict(color=upper_color, width=10),
            name=target_name,
            legendgroup='target',
            showlegend=True
        ))

        # Forearm segment
        fig.add_trace(go.Scatter3d(
            x=[elbow[0], wrist[0]],
            y=[elbow[1], wrist[1]],
            z=[elbow[2], wrist[2]],
            mode='lines',
            line=dict(color=fore_color, width=10),
            legendgroup='target',
            showlegend=False
        ))

        # Joint markers
        fig.add_trace(go.Scatter3d(
            x=[shoulder[0], elbow[0], wrist[0]],
            y=[shoulder[1], elbow[1], wrist[1]],
            z=[shoulder[2], elbow[2], wrist[2]],
            mode='markers',
            marker=dict(
                size=[14, 12, 10],
                color=['#d62728', '#ff7f0e', '#9467bd'],
                symbol='circle',
                opacity=joint_opacity
            ),
            name='Target Joints',
            legendgroup='target',
            showlegend=False,
            text=['Shoulder', 'Target Elbow', 'Target Wrist'],
            hoverinfo='text+x+y+z'
        ))

        # --- USER ARM (Real-time Input) ---
        if user_angles is not None:
            u_shoulder, u_elbow, u_wrist = self.get_joint_positions(user_angles)
            
            # User Upper Arm (Orange)
            fig.add_trace(go.Scatter3d(
                x=[u_shoulder[0], u_elbow[0]],
                y=[u_shoulder[1], u_elbow[1]],
                z=[u_shoulder[2], u_elbow[2]],
                mode='lines+markers',
                line=dict(color='#ff7f0e', width=12), # Orange
                marker=dict(size=4, color='#ff7f0e'),
                name='Your Arm',
                legendgroup='user',
                showlegend=True
            ))

            # User Forearm (Red)
            fig.add_trace(go.Scatter3d(
                x=[u_elbow[0], u_wrist[0]],
                y=[u_elbow[1], u_wrist[1]],
                z=[u_elbow[2], u_wrist[2]],
                mode='lines+markers',
                line=dict(color='#d62728', width=12), # Red
                marker=dict(size=4, color='#d62728'),
                legendgroup='user',
                showlegend=False
            ))
            
            # User Joints (Bright)
            fig.add_trace(go.Scatter3d(
                x=[u_shoulder[0], u_elbow[0], u_wrist[0]],
                y=[u_shoulder[1], u_elbow[1], u_wrist[1]],
                z=[u_shoulder[2], u_elbow[2], u_wrist[2]],
                mode='markers',
                marker=dict(
                    size=[16, 14, 12],
                    color='white',
                    line=dict(color='black', width=2),
                    symbol='diamond'
                ),
                name='User Joints',
                legendgroup='user',
                showlegend=False,
                text=['Shoulder', 'Your Elbow', 'Your Wrist'],
                hoverinfo='text+x+y+z'
            ))

        # Layout configuration
        axis_range = self.total_arm_len * 1.3
        fig.update_layout(
            title=dict(text=title, x=0.5),
            scene=dict(
                xaxis=dict(
                    range=[-axis_range, axis_range],
                    title='X (Forward)',
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    range=[-axis_range, axis_range],
                    title='Y (Left)',
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                zaxis=dict(
                    range=[-axis_range, axis_range],
                    title='Z (Up)',
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8)
                )
            ),
            showlegend=True,
            legend=dict(x=0, y=1),
            height=height,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig

    def create_2d_views(
        self,
        joint_angles: torch.Tensor,
        title: str = "Orthographic Views",
        height: int = 300
    ) -> go.Figure:
        """
        Create 2D orthographic projections (front, side, top).

        Args:
            joint_angles: Shape (4,) tensor in radians
            title: Figure title
            height: Figure height in pixels

        Returns:
            Plotly Figure with 3 subplots
        """
        shoulder, elbow, wrist = self.get_joint_positions(joint_angles)

        # Create positions array for easier plotting
        x = [shoulder[0], elbow[0], wrist[0]]
        y = [shoulder[1], elbow[1], wrist[1]]
        z = [shoulder[2], elbow[2], wrist[2]]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Front View (X-Z)', 'Side View (Y-Z)', 'Top View (X-Y)'],
            horizontal_spacing=0.08
        )

        line_style = dict(color='#1f77b4', width=4)
        marker_style = dict(size=10, color=['#d62728', '#ff7f0e', '#9467bd'])

        # Front view (X-Z plane - looking from +Y)
        fig.add_trace(go.Scatter(
            x=x, y=z,
            mode='lines+markers+text',
            line=line_style,
            marker=marker_style,
            name='Arm',
            text=['Shoulder', 'Elbow', 'Wrist'],
            textposition="top center",
            hoverinfo='text+x+y'
        ), row=1, col=1)

        # Side view (Y-Z plane - looking from -X)
        fig.add_trace(go.Scatter(
            x=y, y=z,
            mode='lines+markers+text',
            line=line_style,
            marker=marker_style,
            showlegend=False,
            text=['Shoulder', 'Elbow', 'Wrist'],
            textposition="top center",
            hoverinfo='text+x+y'
        ), row=1, col=2)

        # Top view (X-Y plane - looking from +Z)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines+markers+text',
            line=line_style,
            marker=marker_style,
            showlegend=False,
            text=['Shoulder', 'Elbow', 'Wrist'],
            textposition="top center",
            hoverinfo='text+x+y'
        ), row=1, col=3)

        # Set equal axes for all subplots
        axis_range = self.total_arm_len * 1.2

        for col in range(1, 4):
            fig.update_xaxes(
                range=[-axis_range, axis_range],
                scaleanchor=f"y{col}" if col == 1 else f"y{col}",
                scaleratio=1,
                showgrid=True,
                gridcolor='lightgray',
                row=1, col=col
            )
            fig.update_yaxes(
                range=[-axis_range, axis_range],
                showgrid=True,
                gridcolor='lightgray',
                row=1, col=col
            )

        # Axis labels (FLU coordinate system: Forward-Left-Up)
        fig.update_xaxes(title_text="X (Forward)", row=1, col=1)
        fig.update_yaxes(title_text="Z (Up)", row=1, col=1)
        fig.update_xaxes(title_text="Y (Left)", row=1, col=2)
        fig.update_yaxes(title_text="Z (Up)", row=1, col=2)
        fig.update_xaxes(title_text="X (Forward)", row=1, col=3)
        fig.update_yaxes(title_text="Y (Left)", row=1, col=3)

        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=height,
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40)
        )

        return fig

    def format_angles_display(
        self,
        joint_angles: torch.Tensor,
        joint_names: Optional[List[str]] = None
    ) -> str:
        """
        Format joint angles for display (convert radians to degrees).

        Args:
            joint_angles: Shape (4,) tensor in radians
            joint_names: Optional joint names (uses defaults if None)

        Returns:
            Formatted multi-line string for display
        """
        names = joint_names or self.DEFAULT_JOINT_NAMES
        angles_deg = torch.rad2deg(joint_angles).cpu().numpy()

        lines = []
        for name, angle in zip(names, angles_deg):
            lines.append(f"{name:>20}: {angle:>7.1f} deg")

        return "\n".join(lines)

    def get_angles_dict(
        self,
        joint_angles: torch.Tensor,
        joint_names: Optional[List[str]] = None
    ) -> dict:
        """
        Get joint angles as a dictionary (degrees).

        Args:
            joint_angles: Shape (4,) tensor in radians
            joint_names: Optional joint names

        Returns:
            Dictionary mapping joint names to angles in degrees
        """
        names = joint_names or self.DEFAULT_JOINT_NAMES
        angles_deg = torch.rad2deg(joint_angles).cpu().numpy()

        return {name: float(angle) for name, angle in zip(names, angles_deg)}
