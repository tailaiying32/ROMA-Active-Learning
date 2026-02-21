"""
ROMA Active Learning - Interactive UI

Main Streamlit application for human-in-the-loop active learning.
Users can provide real-time feasibility assessments for arm poses.

Run with:
    cd /Users/liubodong/dev/Research/ROMA
    streamlit run active_learning/ui/app.py
"""

import sys
import time
import base64
import os
from pathlib import Path

# Add parent directories to path for imports
_this_file = Path(__file__).resolve()
_active_learning_dir = _this_file.parent.parent  # active_learning/
_roma_dir = _active_learning_dir.parent  # ROMA/

if str(_roma_dir) not in sys.path:
    sys.path.insert(0, str(_roma_dir))
if str(_active_learning_dir) not in sys.path:
    sys.path.insert(0, str(_active_learning_dir))

import streamlit as st
import streamlit.components.v1 as components
import torch

from ui.session_manager import (
    initialize_session_state,
    get_session,
    initialize_pipeline,
    request_next_query,
    submit_user_response,
    reset_session,
    get_budget,
    get_progress,
    AppState,
)
from ui.arm_visualizer import ArmVisualizer
from ui.udp_receiver import get_receiver


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="ROMA Active Learning",
        page_icon="💪",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Reduce top margin
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 1rem;
                }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session
    initialize_session_state()
    session = get_session()

    # Start UDP receiver globally
    get_receiver(port=5005)

    # Header
    st.title("ROMA Active Learning")

    # Sidebar
    render_sidebar()

    # Main content based on state
    if session.app_state == AppState.SETUP:
        render_setup_page()

    elif session.app_state == AppState.READY:
        render_ready_state()

    elif session.app_state == AppState.AWAITING_RESPONSE:
        render_query_state()

    elif session.app_state == AppState.COMPLETED:
        render_completed_state()

    elif session.app_state == AppState.ERROR:
        render_error_state()

    # Footer with history
    if session.iteration > 0:
        render_history()


def render_sidebar():
    """Render sidebar with configuration and controls."""
    session = get_session()

    with st.sidebar:
        st.header("Session Info")

        if session.app_state == AppState.SETUP:
            st.info("Configure arm parameters to start")

        elif session.app_state in [AppState.READY, AppState.AWAITING_RESPONSE]:
            # Progress
            budget = get_budget()
            st.metric("Current Query", f"{session.iteration + 1} / {budget}")
            st.progress(get_progress())

            # Stats
            if session.query_history:
                feasible = sum(1 for q in session.query_history if q.outcome > 0.5)
                infeasible = len(session.query_history) - feasible
                col1, col2 = st.columns(2)
                col1.metric("Feasible", feasible)
                col2.metric("Infeasible", infeasible)

            # Arm info
            st.markdown("---")
            st.subheader("Arm Configuration")
            st.text(f"Upper Arm: {session.upper_arm_length:.2f} m")
            st.text(f"Forearm: {session.forearm_length:.2f} m")

            # Reset button
            st.markdown("---")
            
            # Live Mode Toggle (Global)
            st.toggle("Enable Real-time Mirror (UDP)", value=False, key="live_mode_toggle")
            
            if st.button("Reset Session", type="secondary"):
                reset_session()
                st.rerun()

        elif session.app_state == AppState.COMPLETED:
            st.success("Session Complete!")
            budget = get_budget()
            st.metric("Total Queries", f"{session.iteration} / {budget}")

            if st.button("Start New Session"):
                reset_session()
                st.rerun()


def render_setup_page():
    """Render initial setup page for arm parameters."""
    st.header("Setup")
    st.write("Configure your arm measurements before starting the assessment.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Arm Measurements")
        upper_arm = st.number_input(
            "Upper Arm Length (meters)",
            min_value=0.15,
            max_value=0.50,
            value=0.33,
            step=0.01,
            help="Length from shoulder to elbow"
        )

        forearm = st.number_input(
            "Forearm Length (meters)",
            min_value=0.15,
            max_value=0.45,
            value=0.28,
            step=0.01,
            help="Length from elbow to wrist"
        )

    with col2:
        st.subheader("Session Settings")
        budget = st.slider(
            "Query Budget",
            min_value=10,
            max_value=100,
            value=20,
            step=5,
            help="Maximum number of poses to assess"
        )

        st.markdown("---")
        st.markdown("""
        **Instructions:**
        1. Enter your arm measurements
        2. Click "Start Session" to begin
        3. For each pose shown, indicate if you can reach it
        4. The system will learn your reachability over time
        """)

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Start Session", type="primary", use_container_width=True):
            with st.spinner("Initializing pipeline..."):
                success = initialize_pipeline(
                    upper_arm_len=upper_arm,
                    forearm_len=forearm,
                    budget=budget
                )
                if success:
                    st.success("Ready!")
                    st.rerun()
                else:
                    st.error(f"Failed to initialize: {get_session().error_message}")


def render_ready_state():
    """Render ready state - waiting to generate next query."""
    session = get_session()

    st.header(f"Query {session.iteration + 1}")

    budget = get_budget()
    st.progress(get_progress(), text=f"Progress: {session.iteration}/{budget} queries")

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Ready for next pose")
        st.write("Click the button below to see the next arm configuration to assess.")

        if st.button("Get Next Pose", type="primary", use_container_width=True):
            with st.spinner("Computing optimal test..."):
                query = request_next_query()
                if query is not None:
                    st.rerun()
                else:
                    if session.app_state == AppState.COMPLETED:
                        st.rerun()
                    else:
                        st.error("Failed to generate query")


def get_image_data_uri(filepath):
    """Helper to convert local file to Data URI."""
    if not filepath or not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
        ext = filepath.split('.')[-1]
        return f"data:image/{ext};base64,{data}"

def render_fast_canvas(target_coords, user_coords, width=600, height=400, bg_images=None):
    """
    Generates HTML/JS for a fast 2D canvas visualization of the arm.
    target_coords: [shoulder, elbow, wrist] (each is [x,y,z])
    user_coords: [shoulder, elbow, wrist] or None
    bg_images: Dict with keys 'front', 'side', 'top' containing image URLs or Data URIs.
    """
    
    t_sh, t_el, t_wr = target_coords
    u_sh, u_el, u_wr = user_coords if user_coords else (None, None, None)
    
    # Background images (default to null if not provided)
    bg_front = bg_images.get('front') if bg_images else None
    bg_side = bg_images.get('side') if bg_images else None
    bg_top = bg_images.get('top') if bg_images else None
    
    # Simple JSON serialization
    def to_js(arr): return str(list(arr)) if arr is not None else "null"
    def to_js_str(s): return f"'{s}'" if s else "null"
    
    html = f"""
    <div style="display: flex; justify-content: space-around;">
        <canvas id="canvas_front" width="{width//3}" height="{height}" style="border:1px solid #ddd"></canvas>
        <canvas id="canvas_side" width="{width//3}" height="{height}" style="border:1px solid #ddd"></canvas>
        <canvas id="canvas_top" width="{width//3}" height="{height}" style="border:1px solid #ddd"></canvas>
    </div>
    <script>
        const t_sh = {to_js(t_sh)};
        const t_el = {to_js(t_el)};
        const t_wr = {to_js(t_wr)};
        
        const u_sh = {to_js(u_sh)};
        const u_el = {to_js(u_el)};
        const u_wr = {to_js(u_wr)};
        
        const bg_front = {to_js_str(bg_front)};
        const bg_side = {to_js_str(bg_side)};
        const bg_top = {to_js_str(bg_top)};
        
        function drawArm(ctx, s, e, w, color, width, dashed) {{
            if (!s || !e || !w) return;
            
            ctx.beginPath();
            ctx.moveTo(s[0], s[1]);
            ctx.lineTo(e[0], e[1]);
            ctx.lineTo(w[0], w[1]);
            ctx.lineWidth = width;
            ctx.strokeStyle = color;
            if (dashed) ctx.setLineDash([5, 5]);
            else ctx.setLineDash([]);
            ctx.stroke();
            
            // Joints
            ctx.fillStyle = color;
            [s, e, w].forEach(p => {{
                ctx.beginPath();
                ctx.arc(p[0], p[1], width * 0.8, 0, 2 * Math.PI);
                ctx.fill();
            }});
        }}
        
        function renderView(canvasId, idxX, dirX, idxY, dirY, label, imgSrc) {{
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const w = canvas.width;
            const h = canvas.height;
            const scale = 200; // pixels per meter
            const cx = w / 2;
            const cy = h / 2;
            
            // Helper to project 3D point to 2D canvas coords
            // Canvas X = cx + (val * dirX)
            // Canvas Y = cy - (val * dirY)  [Note: Canvas Y is down, so minus moves up]
            const proj = (p) => {{
                if (!p) return null;
                return [
                    cx + (p[idxX] * dirX) * scale, 
                    cy - (p[idxY] * dirY) * scale
                ];
            }};

            // Draw function that runs after image load or immediately
            const drawContent = () => {{
                // Grid (faint on top)
                ctx.strokeStyle = 'rgba(238, 238, 238, 0.5)';
                ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(w, cy); ctx.stroke();
                
                ctx.font = "12px sans-serif";
                ctx.fillStyle = "#888";
                ctx.fillText(label, 10, 20);
                
                // Draw Target (Green, Solid)
                // drawArm(ctx, proj(t_sh), proj(t_el), proj(t_wr), '#1f77b4', 4, true);
                drawArm(ctx, proj(t_sh), proj(t_el), proj(t_wr), '#52a447', 6, true);
                
                // Draw User (Orange, Solid)
                if (u_sh) {{
                    drawArm(ctx, proj(u_sh), proj(u_el), proj(u_wr), '#ff7f0e', 6, false);
                }}
            }};

            // Clear first
            ctx.clearRect(0, 0, w, h);

            if (imgSrc && imgSrc !== 'null') {{
                const img = new Image();
                img.onload = () => {{
                    const imgAspect = img.width / img.height;
                    const canvasAspect = w / h;
                    let dw, dh;
                    if (imgAspect > canvasAspect) {{
                        dw = w;
                        dh = w / imgAspect;
                    }} else {{
                        dh = h;
                        dw = h * imgAspect;
                    }}

                    if (canvasId === 'canvas_top') {{
                        // Rotate 90° CCW, zoom out, align right shoulder with origin
                        const rotAspect = img.height / img.width;
                        if (rotAspect > canvasAspect) {{
                            dw = w;
                            dh = w / rotAspect;
                        }} else {{
                            dh = h;
                            dw = h * rotAspect;
                        }}
                        const rw = dh;
                        const rh = dw;
                        const zoom = 0.60;
                        const shX = 0.55;
                        const shY = 0.85;
                        const zrw = rw * zoom;
                        const zrh = rh * zoom;
                        ctx.save();
                        ctx.translate(cx, cy);
                        ctx.rotate(-Math.PI / 2);
                        ctx.drawImage(img, -shX * zrw, -shY * zrh, zrw, zrh);
                        ctx.restore();
                    }} else if (canvasId === 'canvas_front') {{
                        // Shift so right shoulder aligns with origin (canvas center)
                        // and zoom out for better fit
                        const shX = 0.78;
                        const shY = 0.85;
                        const zoom = 0.65;
                        dw *= zoom;
                        dh *= zoom;
                        ctx.drawImage(img, cx - shX * dw, cy - shY * dh, dw, dh);
                    }} else if (canvasId === 'canvas_side') {{
                        // Side view: zoom in for better visibility
                        const shX = 0.55;
                        const shY = 0.50;
                        const zoom = 0.65; 
                        dw *= zoom;
                        dh *= zoom;
                        ctx.drawImage(img, cx - shX * dw, cy - shY * dh, dw, dh);
                    }} else {{
                        ctx.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
                    }}
                    drawContent();
                }};
                img.src = imgSrc;
            }} else {{
                // Procedural Fallback (The Gray Cutouts)
                ctx.fillStyle = "#e0e0e0";
                ctx.strokeStyle = "#cccccc";
                
                if (label.includes("Behind")) {{
                    // Behind View (-Y=Right, Z=Up) - Looking from behind user
                    // Origin is Right Shoulder (Y=0)
                    // Body extends to the Left (+Y direction)
                    // Head
                    let head = proj([0, 0.2, 0.25]); // Y=0.2 (Left of shoulder), Z=0.25 (Up)
                    ctx.beginPath();
                    ctx.arc(head[0], head[1], 0.1 * scale, 0, 2 * Math.PI);
                    ctx.fill(); 
                    
                    // Torso (extends left from right shoulder)
                    let tl = proj([0, 0.05, 0.15]);   // Right edge
                    let br = proj([0, 0.35, -0.4]);   // Left edge, bottom
                    ctx.fillRect(tl[0], tl[1], (br[0]-tl[0]), (br[1]-tl[1]));

                    // Left Shoulder marker
                    let ls = proj([0, 0.4, 0]);
                    ctx.beginPath(); ctx.arc(ls[0], ls[1], 0.05*scale, 0, 2*Math.PI); ctx.fill();

                }} else if (label.includes("Left Side")) {{
                    // Left Side View (-X=Back, Z=Up) - Looking from left side
                    // Facing direction is +X (forward), appears as depth
                    // Head
                    let head = proj([0, 0, 0.25]); // X=0 (at shoulder), Z=0.25 (up)
                    ctx.beginPath();
                    ctx.arc(head[0], head[1], 0.1 * scale, 0, 2 * Math.PI);
                    ctx.fill();
                    
                    // Torso (Profile view from left side)
                    let p1 = proj([-0.1, 0, 0.15]);   // Back edge, top
                    let p2 = proj([0.1, 0, -0.4]);     // Front edge, bottom
                    let rx = Math.min(p1[0], p2[0]);
                    let ry = Math.min(p1[1], p2[1]);
                    let rw = Math.abs(p1[0] - p2[0]);
                    let rh = Math.abs(p1[1] - p2[1]);
                    ctx.fillRect(rx, ry, rw, rh);

                    // Nose pointing forward (+X direction, appears left in view)
                    let noseBase = proj([0.08, 0, 0.25]);
                    let noseTip = proj([0.2, 0, 0.25]);
                    ctx.beginPath();
                    ctx.moveTo(noseBase[0], noseBase[1]-5);
                    ctx.lineTo(noseBase[0], noseBase[1]+5);
                    ctx.lineTo(noseTip[0], noseTip[1]);
                    ctx.fill();

                }} else if (label.includes("Top")) {{
                    // Top View (-Y=Right, X=Front) - Looking from above
                    // Origin is Right Shoulder (Y=0, facing +X forward)
                    // Body extends left (+Y direction)
                    // Head
                    let head = proj([0, 0.2, 0]); // X=0, Y=0.2 (left of shoulder)
                    ctx.beginPath();
                    ctx.arc(head[0], head[1], 0.1 * scale, 0, 2 * Math.PI);
                    ctx.fill();
                    
                    // Shoulders (Oval) - extends left from right shoulder
                    // Center at Y=0.2 (left of right shoulder), width along Y, depth along X
                    let s_cen = proj([0, 0.2, 0]);
                    ctx.beginPath();
                    ctx.ellipse(s_cen[0], s_cen[1], 0.2 * scale, 0.1 * scale, 0, 0, 2 * Math.PI);
                    ctx.fill();

                    // Nose pointing forward (+X, appears up in view)
                    let noseBase = proj([0.08, 0.2, 0]);
                    let noseTip = proj([0.2, 0.2, 0]);
                    ctx.beginPath();
                    ctx.moveTo(noseBase[0]-5, noseBase[1]);
                    ctx.lineTo(noseBase[0]+5, noseBase[1]);
                    ctx.lineTo(noseTip[0], noseTip[1]);
                    ctx.fill();
                }}
                
                drawContent();
            }}
        }}
        
        // Behind View: Looking from behind user (Y=-Right, Z=Up)
        renderView('canvas_front', 1, -1, 2, 1, "Behind: Right \u2192, Up \u2191", bg_front);

        // Left Side View: Looking from left side (X=-Back, Z=Up)
        renderView('canvas_side', 0, -1, 2, 1, "Left Side: Back \u2192, Up \u2191", bg_side);

        // Top View: Looking from above (Y=-Right, X=Front)
        renderView('canvas_top', 1, -1, 0, 1, "Top: Right \u2192, Front \u2191", bg_top);
    </script>
    """
    return html

def render_query_state():
    """Render query state - showing visualization, awaiting yes/no."""
    session = get_session()
    receiver = get_receiver()

    # Check Global Live Mode
    live_mode = st.session_state.get("live_mode_toggle", False)
    
    # Fetch live data
    user_angles = None
    if live_mode:
        user_angles = receiver.get_latest_angles()

    # Create visualizer logic
    visualizer = ArmVisualizer(
        upper_arm_length=session.upper_arm_length,
        forearm_length=session.forearm_length
    )

    # Layout: Response buttons
    col_resp1, col_resp2, col_resp3 = st.columns([1, 1, 1])

    with col_resp1:
        if st.button("✓ Yes, I can reach this", type="primary", use_container_width=True):
            submit_user_response(True)
            st.rerun()

    with col_resp2:
        if st.button("✗ No, I cannot reach this", type="secondary", use_container_width=True):
            submit_user_response(False)
            st.rerun()
    
    # Visualization Area
    st.markdown("""
        <h4 style='text-align: center; margin-bottom: 10px;'>
            <span style='color: #1f77b4;'>Target (Ghost)</span> 
            vs 
            <span style='color: #ff7f0e;'>You (Solid)</span>
        </h4>
    """, unsafe_allow_html=True)
    
    # Pre-calculate coordinates for the Canvas
    # Target
    t_s, t_e, t_w = visualizer.get_joint_positions(session.current_query)
    target_coords = [t_s.tolist(), t_e.tolist(), t_w.tolist()]
    
    # User
    user_coords = None
    if user_angles is not None:
        u_s, u_e, u_w = visualizer.get_joint_positions(user_angles)
        user_coords = [u_s.tolist(), u_e.tolist(), u_w.tolist()]
    
    # Background Images Config
    # TODO: Replace None with paths to your images, e.g., "images/front_view.png"
    # Or use URLs. 
    _ui_dir = os.path.dirname(os.path.abspath(__file__))
    BACKGROUND_IMAGES = {
        'front': get_image_data_uri(os.path.join(_ui_dir, "imgs", "front.png")),
        'side': get_image_data_uri(os.path.join(_ui_dir, "imgs", "side.png")),
        'top': get_image_data_uri(os.path.join(_ui_dir, "imgs", "top.png")),
    }
    
    # Render HTML Canvas (Fast)
    canvas_html = render_fast_canvas(target_coords, user_coords, width=900, height=300, bg_images=BACKGROUND_IMAGES)
    components.html(canvas_html, height=320)

    # Display joint angles (Static info)
    with st.expander("Target Joint Angles (degrees)", expanded=False):
        joint_names = session.config.get('prior', {}).get('joint_names', None)
        angles_text = visualizer.format_angles_display(
            session.current_query,
            joint_names
        )
        st.code(angles_text)

    # Live Mode Loop
    if live_mode:
        time.sleep(0.02) # Increased refresh rate for smoother animation
        st.rerun()


def render_completed_state():
    """Render completed state - session finished."""
    session = get_session()

    st.header("Session Complete!")
    
    st.success(f"You completed {session.iteration} pose assessments.")

    # Summary statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Queries", session.iteration)

    with col2:
        feasible = sum(1 for q in session.query_history if q.outcome > 0.5)
        st.metric("Feasible Poses", feasible)

    with col3:
        infeasible = len(session.query_history) - feasible
        st.metric("Infeasible Poses", infeasible)

    # Ratio visualization
    if session.query_history:
        feasible_ratio = feasible / len(session.query_history)
        st.progress(feasible_ratio, text=f"Feasible: {feasible_ratio*100:.1f}%")

    st.markdown("---")

    # Actions
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start New Session", type="primary", use_container_width=True):
            reset_session()
            st.rerun()

    with col2:
        if st.button("Export Results", use_container_width=True):
            st.info("Export functionality coming soon")


def render_error_state():
    """Render error state."""
    session = get_session()

    st.error("An error occurred")
    st.code(session.error_message)

    if st.button("Reset and Try Again"):
        reset_session()
        st.rerun()


def render_history():
    """Render query history at the bottom of the page."""
    session = get_session()

    if not session.query_history:
        return

    st.markdown("---")

    with st.expander(f"Query History ({len(session.query_history)} responses)", expanded=False):
        # Create a simple table
        for i, record in enumerate(reversed(session.query_history)):
            col1, col2, col3 = st.columns([1, 3, 1])

            with col1:
                st.write(f"**#{record.iteration + 1}**")

            with col2:
                angles = torch.rad2deg(record.test_point).cpu().numpy()
                st.caption(
                    f"HAA:{angles[0]:.0f}° FE:{angles[1]:.0f}° ROT:{angles[2]:.0f}° Elb:{angles[3]:.0f}°"
                )

            with col3:
                if record.outcome > 0.5:
                    st.success("Feasible")
                else:
                    st.error("Infeasible")

            if i < len(session.query_history) - 1:
                st.divider()


if __name__ == "__main__":
    main()