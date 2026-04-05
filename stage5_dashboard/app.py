"""
app.py
------
Stage 5 — Streamlit Dashboard for Colorectal Cancer CAD System.

Features:
    - Upload H&E histopathology patch
    - Real-time cancer detection (colon_aca vs colon_n)
    - Grad-CAM heatmap visualization
    - Confidence scores and risk assessment
    - Model information panel

Run:
    streamlit run stage5_dashboard/app.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'stage3_classification'))

from stage5_dashboard.gradcam import GradCAM
from stage5_dashboard.utils import (
    CLASS_COLORS,
    CLASS_ICONS,
    CLASS_LABELS,
    CLASS_NAMES,
    format_confidence,
    get_risk_level,
    load_model,
    predict,
    preprocess_image,
    validate_image,
)

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="CRC-CAD | Colorectal Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS — Clinical dark theme
# ------------------------------------------------------------------

st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    /* Root variables */
    :root {
        --bg-primary   : #0a0e1a;
        --bg-secondary : #111827;
        --bg-card      : #1a2234;
        --accent-blue  : #3b82f6;
        --accent-green : #10b981;
        --accent-red   : #ef4444;
        --accent-amber : #f59e0b;
        --text-primary : #f1f5f9;
        --text-muted   : #94a3b8;
        --border       : #1e293b;
    }

    /* Global */
    .stApp {
        background-color: var(--bg-primary);
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }

    /* Cards */
    .cad-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }

    /* Result box — cancer */
    .result-cancer {
        background: linear-gradient(135deg, #1a0a0a, #2d1515);
        border: 2px solid var(--accent-red);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    /* Result box — normal */
    .result-normal {
        background: linear-gradient(135deg, #0a1a0f, #0f2d1a);
        border: 2px solid var(--accent-green);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border-bottom: 1px solid var(--border);
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        border-radius: 0 0 16px 16px;
    }

    .app-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin: 0;
    }

    .app-subtitle {
        color: var(--text-muted);
        font-size: 0.9rem;
        margin-top: 0.25rem;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Metric */
    .metric-box {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--accent-blue);
    }

    .metric-label {
        color: var(--text-muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.25rem;
    }

    /* Confidence bar */
    .conf-bar-container {
        background: #1e293b;
        border-radius: 999px;
        height: 8px;
        width: 100%;
        margin: 0.5rem 0;
    }

    .conf-bar-fill {
        height: 8px;
        border-radius: 999px;
        transition: width 0.5s ease;
    }

    /* Risk badge */
    .risk-high   { background:#ef444420; color:#ef4444; border:1px solid #ef4444; padding:4px 12px; border-radius:999px; font-size:0.8rem; font-family:'IBM Plex Mono',monospace; }
    .risk-mod    { background:#f59e0b20; color:#f59e0b; border:1px solid #f59e0b; padding:4px 12px; border-radius:999px; font-size:0.8rem; font-family:'IBM Plex Mono',monospace; }
    .risk-low    { background:#10b98120; color:#10b981; border:1px solid #10b981; padding:4px 12px; border-radius:999px; font-size:0.8rem; font-family:'IBM Plex Mono',monospace; }
    .risk-vlow   { background:#3b82f620; color:#3b82f6; border:1px solid #3b82f6; padding:4px 12px; border-radius:999px; font-size:0.8rem; font-family:'IBM Plex Mono',monospace; }

    /* Section label */
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }

    /* Disclaimer */
    .disclaimer {
        background: #1e293b;
        border-left: 3px solid var(--accent-amber);
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 1rem;
    }

    /* Streamlit overrides */
    .stButton > button {
        background: var(--accent-blue);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #2563eb;
        transform: translateY(-1px);
    }

    h1, h2, h3 {
        color: var(--text-primary) !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    p, li { color: var(--text-muted); }

    .stFileUploader {
        background: var(--bg-card);
        border: 2px dashed var(--border);
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Model loading (cached)
# ------------------------------------------------------------------

CHECKPOINT_PATH = ROOT / 'stage3_classification' / 'checkpoints' / 'efficientnet_b0_colon.pth'


@st.cache_resource(show_spinner=False)
def get_model():
    """Load model once and cache for session."""
    try:
        model, checkpoint = load_model(str(CHECKPOINT_PATH), device='cpu')
        return model, checkpoint, None
    except FileNotFoundError as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, f"Error loading model: {e}"


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------

st.markdown("""
<div class="app-header">
    <p class="app-title">🔬 CRC-CAD</p>
    <p class="app-subtitle">Colorectal Cancer · Computer-Aided Detection System · EfficientNet-B0</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ System")

    model, checkpoint, load_error = get_model()

    if model is not None:
        epoch    = checkpoint.get('epoch', 'N/A')
        val_acc  = checkpoint.get('val_acc', 0)
        st.success(f"Model loaded ✓")
        st.markdown(f"""
        <div class="cad-card">
            <div class="section-label">Model Info</div>
            <p style="margin:0;color:#f1f5f9;font-family:'IBM Plex Mono',monospace;font-size:0.85rem;">
                EfficientNet-B0<br>
                Best epoch: {epoch}<br>
                Val accuracy: {val_acc*100:.2f}%<br>
                Classes: 2 (binary)
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Model not loaded")
        if load_error:
            st.caption(load_error)
        st.info(
            "Place `efficientnet_b0_colon.pth` in:\n"
            "`stage3_classification/checkpoints/`"
        )

    st.markdown("---")
    st.markdown("### 🎨 Visualization")
    gradcam_alpha = st.slider(
        "Grad-CAM opacity",
        min_value=0.1, max_value=0.9,
        value=0.4, step=0.1,
        help="Controls how transparent the heatmap overlay is"
    )
    colormap = st.selectbox(
        "Heatmap colormap",
        options=['jet', 'hot', 'inferno', 'plasma', 'viridis'],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    <p style="font-size:0.8rem;">
    This CAD system detects colorectal adenocarcinoma from H&E
    histopathology patches. Built as part of an ICMR-aligned
    research pipeline.
    </p>
    <p style="font-size:0.8rem;">
    <b>Model:</b> EfficientNet-B0<br>
    <b>Dataset:</b> LC25000 colon subset<br>
    <b>Test accuracy:</b> 99.93%<br>
    <b>F1 Score:</b> 0.9993
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
    ⚠️ For research purposes only. Not for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------
# Main content
# ------------------------------------------------------------------

if model is None:
    st.error(f"Cannot run analysis: {load_error}")
    st.info(
        "**Setup required:**\n"
        "1. Train the model using `notebooks/03_model_training.ipynb`\n"
        "2. Download `efficientnet_b0_colon.pth` from Google Drive\n"
        "3. Place in `stage3_classification/checkpoints/`\n"
        "4. Restart this app"
    )
    st.stop()

# Upload section
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown('<div class="section-label">Upload H&E Patch</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Upload H&E histopathology patch",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Upload a 224×224 or larger H&E stained histopathology image patch",
        label_visibility="collapsed",
    )

with col_info:
    st.markdown("""
    <div class="cad-card">
        <div class="section-label">Supported Inputs</div>
        <p style="font-size:0.85rem;margin:0;">
            • PNG, JPG, JPEG<br>
            • TIF, TIFF<br>
            • Min: 32×32 px<br>
            • H&E stained tissue<br>
            • Colon tissue patches
        </p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    is_valid, msg = validate_image(image)
    if not is_valid:
        st.error(f"Invalid image: {msg}")
        st.stop()

    st.markdown("---")

    # Run analysis
    with st.spinner("Analyzing tissue patch..."):
        tensor = preprocess_image(image)

        # Prediction
        result = predict(model, tensor, device='cpu')

        # Grad-CAM
        try:
            cam       = GradCAM(model)
            heatmap, overlay_img, _ = cam.run(
                tensor, image,
                class_idx=result['class_idx'],
                alpha=gradcam_alpha,
            )
            cam.remove_hooks()
            gradcam_available = True
        except Exception as e:
            gradcam_available = False
            gradcam_error     = str(e)

    # ------ Results layout ------
    col_img, col_cam, col_result = st.columns([1, 1, 1])

    with col_img:
        st.markdown('<div class="section-label">Original Patch</div>', unsafe_allow_html=True)
        display_img = image.resize((400, 400), Image.LANCZOS)
        st.image(display_img, use_container_width=True)
        w, h = image.size
        st.caption(f"Size: {w}×{h}px")

    with col_cam:
        st.markdown('<div class="section-label">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
        if gradcam_available:
            overlay_display = overlay_img.resize((400, 400), Image.LANCZOS)
            st.image(overlay_display, use_container_width=True)
            st.caption("Red regions = model attention areas")
        else:
            st.warning(f"Grad-CAM unavailable: {gradcam_error}")
            st.image(display_img, use_container_width=True)

    with col_result:
        st.markdown('<div class="section-label">Detection Result</div>', unsafe_allow_html=True)

        # Main result box
        result_class = "result-cancer" if result['is_cancer'] else "result-normal"
        icon         = result['icon']
        label        = result['label']
        conf_pct     = format_confidence(result['confidence'])
        color        = result['color']

        st.markdown(f"""
        <div class="{result_class}">
            <div style="font-size:2.5rem;">{icon}</div>
            <div style="font-size:1.1rem;font-weight:600;color:{color};margin:0.5rem 0;">
                {label}
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;color:{color};font-weight:700;">
                {conf_pct}
            </div>
            <div style="color:#94a3b8;font-size:0.8rem;">confidence</div>
        </div>
        """, unsafe_allow_html=True)

        # Risk level
        risk = get_risk_level(result['confidence'], result['is_cancer'])
        risk_class = {
            'High': 'risk-high',
            'Moderate': 'risk-mod',
            'Low': 'risk-low',
            'Very Low': 'risk-vlow',
        }[risk]

        st.markdown(f"""
        <div style="margin-top:1rem;">
            <div class="section-label">Risk Level</div>
            <span class="{risk_class}">{risk}</span>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown("""
        <div style="margin-top:1.5rem;">
            <div class="section-label">Class Probabilities</div>
        </div>
        """, unsafe_allow_html=True)

        for cls_name, prob in result['probabilities'].items():
            bar_color = CLASS_COLORS[cls_name]
            bar_width = int(prob * 100)
            lbl       = CLASS_LABELS[cls_name]
            prob_pct  = f"{prob*100:.1f}%"

            st.markdown(f"""
            <div style="margin-bottom:0.75rem;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="font-size:0.8rem;color:#94a3b8;">{lbl}</span>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;color:{bar_color};">
                        {prob_pct}
                    </span>
                </div>
                <div class="conf-bar-container">
                    <div class="conf-bar-fill"
                         style="width:{bar_width}%;background:{bar_color};">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ------ Detailed metrics ------
    st.markdown("---")
    st.markdown('<div class="section-label">Analysis Details</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{format_confidence(result['confidence'])}</div>
            <div class="metric-label">Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{result['class_name']}</div>
            <div class="metric-label">Predicted Class</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{risk}</div>
            <div class="metric-label">Risk Level</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        img_w, img_h = image.size
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{img_w}×{img_h}</div>
            <div class="metric-label">Image Size (px)</div>
        </div>
        """, unsafe_allow_html=True)

    # ------ Download results ------
    st.markdown("---")
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        if gradcam_available:
            import io
            buf = io.BytesIO()
            overlay_img.save(buf, format='PNG')
            st.download_button(
                label="⬇ Download Grad-CAM overlay",
                data=buf.getvalue(),
                file_name=f"gradcam_{result['class_name']}.png",
                mime="image/png",
            )

    with dl_col2:
        import json
        report = {
            'filename'     : uploaded_file.name,
            'prediction'   : result['class_name'],
            'label'        : result['label'],
            'confidence'   : result['confidence'],
            'risk_level'   : risk,
            'probabilities': result['probabilities'],
            'model'        : 'EfficientNet-B0',
            'note'         : 'Research use only. Not for clinical diagnosis.',
        }
        st.download_button(
            label="⬇ Download JSON report",
            data=json.dumps(report, indent=2),
            file_name=f"cad_report_{result['class_name']}.json",
            mime="application/json",
        )

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;">
        <div style="font-size:4rem;margin-bottom:1rem;">🔬</div>
        <h3 style="color:#f1f5f9;">Upload an H&E patch to begin analysis</h3>
        <p style="color:#64748b;max-width:500px;margin:0 auto;">
            Upload a hematoxylin and eosin stained histopathology patch
            from colon tissue. The system will classify it as adenocarcinoma
            (cancer) or benign tissue, and generate a Grad-CAM attention map.
        </p>
        <br>
        <p style="color:#475569;font-size:0.85rem;">
            Supported formats: PNG · JPG · JPEG · TIF · TIFF
        </p>
    </div>
    """, unsafe_allow_html=True)
