from pathlib import Path
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image

from src.visioninspect.inference import VisionInspectInference
from src.visioninspect.video import inspect_video


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VisionInspect AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL UI
# ============================================================

st.markdown(
    """
<style>

[data-testid="stHeader"] {
    height: 0px;
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.stApp {
    background:
        radial-gradient(
            circle at 90% 0%,
            rgba(37, 99, 235, 0.17),
            transparent 28%
        ),
        radial-gradient(
            circle at 20% 100%,
            rgba(6, 182, 212, 0.08),
            transparent 30%
        ),
        #07111f;

    color: #e6edf7 !important;
}

.block-container {
    max-width: 1480px;
    padding-top: 0.65rem !important;
    padding-bottom: 0.75rem !important;
    padding-left: 1.7rem !important;
    padding-right: 1.7rem !important;
}

.stApp p,
.stApp label,
.stApp span,
.stApp div,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp li {
    color: #e6edf7;
}

[data-testid="stMarkdownContainer"] p {
    color: #cbd5e1 !important;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}


/* ==========================================================
   SIDEBAR
========================================================== */

section[data-testid="stSidebar"] {
    width: 250px !important;
    min-width: 250px !important;

    background:
        linear-gradient(
            180deg,
            #081426 0%,
            #0b1930 100%
        );

    border-right:
        1px solid rgba(148, 163, 184, 0.14);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

.sidebar-brand {
    font-size: 19px;
    font-weight: 800;
    color: white !important;
    margin-bottom: 2px;
}

.sidebar-caption {
    color: #64748b !important;
    font-size: 11px;
}


/* ==========================================================
   HERO
========================================================== */

.hero {
    min-height: 112px;
    padding: 20px 28px;
    border-radius: 18px;

    background:
        linear-gradient(
            120deg,
            #312e81 0%,
            #1d4ed8 52%,
            #0891b2 100%
        );

    box-shadow:
        0 12px 35px rgba(0, 0, 0, 0.25);

    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "";
    position: absolute;

    width: 220px;
    height: 220px;

    border-radius: 50%;

    background:
        rgba(255, 255, 255, 0.055);

    right: -70px;
    top: -100px;
}

.hero-top {
    color: #bae6fd !important;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 6px;
}

.hero-title {
    color: white !important;
    font-size: 32px;
    font-weight: 850;
    line-height: 1.05;
    margin: 0;
}

.hero-description {
    color: #dbeafe !important;
    font-size: 13px;
    margin-top: 8px;
    max-width: 900px;
    line-height: 1.45;
}


/* ==========================================================
   TABS
========================================================== */

.stTabs [data-baseweb="tab-list"] {
    gap: 22px;
    height: 42px;
}

.stTabs [data-baseweb="tab"] {
    padding-top: 7px;
    padding-bottom: 7px;
    font-size: 13px;
    font-weight: 600;
}


/* ==========================================================
   UPLOADER
========================================================== */

[data-testid="stFileUploader"] {
    background:
        rgba(15, 23, 42, 0.72);

    border:
        1px solid rgba(148, 163, 184, 0.15);

    border-radius: 12px;
    padding: 4px 8px;
}


/* ==========================================================
   RESULT SUMMARY
========================================================== */

.summary-grid {
    display: grid;

    grid-template-columns:
        1.55fr 1fr 1fr 1fr;

    gap: 10px;

    margin-top: 8px;
    margin-bottom: 10px;
}

.summary-card {
    min-height: 82px;
    padding: 13px 16px;
    border-radius: 14px;

    background:
        linear-gradient(
            145deg,
            rgba(22, 34, 57, 0.97),
            rgba(12, 23, 42, 0.97)
        );

    border:
        1px solid rgba(99, 102, 241, 0.22);
}

.summary-label {
    color: #94a3b8 !important;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 4px;
}

.summary-value {
    color: #f8fafc !important;
    font-size: 22px;
    font-weight: 750;
}

.status-normal {
    background:
        linear-gradient(
            135deg,
            rgba(5, 150, 105, 0.27),
            rgba(6, 78, 59, 0.18)
        );

    border:
        1px solid rgba(52, 211, 153, 0.48);
}

.status-normal .summary-value {
    color: #6ee7b7 !important;
}

.status-anomaly {
    background:
        linear-gradient(
            135deg,
            rgba(220, 38, 38, 0.27),
            rgba(127, 29, 29, 0.18)
        );

    border:
        1px solid rgba(248, 113, 113, 0.48);
}

.status-anomaly .summary-value {
    color: #fca5a5 !important;
}

.status-note {
    color: #94a3b8 !important;
    font-size: 10px;
    margin-top: 2px;
}


/* ==========================================================
   HEADINGS
========================================================== */

.section-title {
    color: #f8fafc !important;
    font-size: 18px;
    font-weight: 760;
    margin-top: 4px;
    margin-bottom: 6px;
}

.image-heading {
    color: #cbd5e1 !important;
    font-size: 12px;
    font-weight: 650;
    margin-bottom: 4px;
}


/* ==========================================================
   BADGES
========================================================== */

.badge {
    display: inline-block;

    padding: 4px 9px;
    margin-right: 4px;

    border-radius: 16px;

    background:
        rgba(59, 130, 246, 0.12);

    border:
        1px solid rgba(96, 165, 250, 0.25);

    color: #bfdbfe !important;
    font-size: 10px;
}


/* ==========================================================
   METRICS
========================================================== */

[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            rgba(22, 34, 57, 0.97),
            rgba(12, 23, 42, 0.97)
        );

    border:
        1px solid rgba(99, 102, 241, 0.20);

    border-radius: 14px;

    padding: 12px 14px;
}


/* ==========================================================
   DIVIDERS / SPACING
========================================================== */

hr {
    margin-top: 8px !important;
    margin-bottom: 8px !important;

    border-color:
        rgba(148, 163, 184, 0.13) !important;
}

[data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}

div[data-testid="stExpander"] {
    border:
        1px solid rgba(148, 163, 184, 0.15);

    border-radius: 10px;
}


/* ==========================================================
   BUTTONS
========================================================== */

.stButton > button {
    border-radius: 10px;

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #0891b2
        );

    color: white !important;
    border: none;
    font-weight: 700;
}


/* ==========================================================
   FOOTER
========================================================== */

.app-footer {
    color: #475569 !important;
    text-align: center;
    font-size: 9px;
    margin-top: 4px;
}


@media (max-width: 1100px) {

    .summary-grid {
        grid-template-columns: 1fr 1fr;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_CONFIGS = {

    "Bottle": {
        "model_path":
            "models/patchcore_bottle_memory.pt",

        "threshold":
            0.629979,

        "threshold_method":
            "Normal validation mean + 3σ",

        "image_auroc":
            1.0000,

        "pixel_auroc":
            0.9819,

        "icon":
            "🍾",
    },

    "Hazelnut": {
        "model_path":
            "models/patchcore_hazelnut_memory.pt",

        "threshold":
            0.774364,

        "threshold_method":
            "Normal validation mean + 3σ",

        "image_auroc":
            1.0000,

        "pixel_auroc":
            0.9841,

        "icon":
            "🌰",
    },

    "Tile": {
        "model_path":
            "models/patchcore_tile_memory.pt",

        "threshold":
            0.669720,

        "threshold_method":
            "Maximum normal validation score",

        "image_auroc":
            1.0000,

        "pixel_auroc":
            0.9428,

        "icon":
            "🧱",
    },
}


# ============================================================
# MODEL CACHE
# ============================================================

@st.cache_resource
def load_engine(
    model_path,
    threshold,
):

    return VisionInspectInference(
        model_path=model_path,
        threshold=threshold,
        device="cpu",
    )


# ============================================================
# RESULTS CACHE
# ============================================================

@st.cache_data
def load_results_csv(
    path,
):

    path = Path(
        path
    )

    if not path.exists():
        return None

    return pd.read_csv(
        path
    )


def find_column(
    dataframe,
    possible_names,
):

    for name in possible_names:

        if name in dataframe.columns:
            return name

    return None


# ============================================================
# VISUAL HELPERS
# ============================================================

def make_heatmap(
    heatmap,
):

    cmap = plt.get_cmap(
        "turbo"
    )

    colored = cmap(
        heatmap
    )[:, :, :3]

    return (
        colored * 255
    ).astype(
        np.uint8
    )


def make_overlay(
    image,
    heatmap,
):

    base = (
        image * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    colored = make_heatmap(
        heatmap
    )

    overlay = (
        0.72 * base
        +
        0.28 * colored
    )

    return overlay.clip(
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">

<div class="hero-top">
AI-POWERED INDUSTRIAL VISUAL INSPECTION
</div>

<div class="hero-title">
🔍 VisionInspect
</div>

<div class="hero-description">
Unsupervised anomaly detection and defect localization
using PatchCore-style deep feature matching.
Image and video inspection with category-specific calibration.
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
<div class="sidebar-brand">
🔍 VisionInspect
</div>

<div class="sidebar-caption">
Industrial Inspection Console
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    selected_category = st.selectbox(
        "Category",
        list(
            MODEL_CONFIGS.keys()
        ),
    )

    inspection_mode = st.radio(
        "Mode",
        [
            "Image",
            "Video",
        ],
        horizontal=True,
    )

    config = MODEL_CONFIGS[
        selected_category
    ]

    st.markdown(
        f"""
<span class="badge">
{config["icon"]} {selected_category}
</span>

<span class="badge">
{inspection_mode}
</span>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption(
        "MODEL CONFIGURATION"
    )

    st.write(
        "**WideResNet50-2**"
    )

    st.caption(
        "256 × 256 input"
    )

    st.caption(
        "128-D embeddings"
    )

    st.caption(
        "500 patch memory bank"
    )

    st.caption(
        f"Threshold: "
        f"{config['threshold']:.4f}"
    )

    st.divider()

    st.caption(
        "DEVELOPMENT BENCHMARK"
    )

    # FIXED SIDEBAR BENCHMARK
    st.markdown(
        f"""
<div style="
    display:flex;
    gap:8px;
    margin-top:6px;
">

<div style="
    flex:1;
    min-width:0;
    padding:10px;
    border-radius:10px;
    background:#101d33;
    border:1px solid #1e3a5f;
">

<div style="
    font-size:9px;
    color:#94a3b8;
">
IMAGE AUROC
</div>

<div style="
    font-size:18px;
    font-weight:700;
    color:#f8fafc;
">
{config['image_auroc']:.4f}
</div>

</div>

<div style="
    flex:1;
    min-width:0;
    padding:10px;
    border-radius:10px;
    background:#101d33;
    border:1px solid #1e3a5f;
">

<div style="
    font-size:9px;
    color:#94a3b8;
">
PIXEL AUROC
</div>

<div style="
    font-size:18px;
    font-weight:700;
    color:#f8fafc;
">
{config['pixel_auroc']:.4f}
</div>

</div>

</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD MODEL
# ============================================================

model_path = Path(
    config["model_path"]
)

if not model_path.exists():

    st.error(
        f"Model not found: {model_path}"
    )

    st.stop()


engine = load_engine(
    str(model_path),
    config["threshold"],
)


# ============================================================
# TABS
# ============================================================

inspection_tab, benchmark_tab, architecture_tab = (
    st.tabs(
        [
            "🔬 Inspection",
            "📊 Benchmark",
            "🧠 Architecture",
        ]
    )
)


# ============================================================
# INSPECTION TAB
# ============================================================

with inspection_tab:

    # ========================================================
    # IMAGE MODE
    # ========================================================

    if inspection_mode == "Image":

        header_col, badge_col = (
            st.columns(
                [4, 1]
            )
        )

        with header_col:

            st.markdown(
                f"""
<div class="section-title">
{config["icon"]} {selected_category} Image Inspection
</div>
""",
                unsafe_allow_html=True,
            )

        with badge_col:

            st.caption(
                "PatchCore • 256 × 256"
            )


        uploaded_file = st.file_uploader(
            "Upload inspection image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "jfif",
            ],
            key="image_upload",
        )


        if uploaded_file is None:

            st.info(
                "Upload an image to begin inspection."
            )


        else:

            image = Image.open(
                uploaded_file
            ).convert(
                "RGB"
            )

            with st.spinner(
                "Analyzing visual features..."
            ):

                result = engine.predict_image(
                    image
                )


            score = float(
                result["score"]
            )

            threshold = float(
                result["threshold"]
            )

            heatmap = result[
                "heatmap"
            ]

            display_image = result[
                "image"
            ]

            difference = (
                score - threshold
            )


            if score >= threshold:

                prediction = "ANOMALOUS"

                status_class = (
                    "status-anomaly"
                )

                status_icon = "⚠️"

                note = (
                    "Visual deviation detected"
                )

            else:

                prediction = "NORMAL"

                status_class = (
                    "status-normal"
                )

                status_icon = "✅"

                note = (
                    "Within learned normal distribution"
                )


            # =================================================
            # RESULT SUMMARY
            # =================================================

            st.markdown(
                f"""
<div class="summary-grid">

<div class="summary-card {status_class}">
<div class="summary-label">
Inspection Result
</div>

<div class="summary-value">
{status_icon} {prediction}
</div>

<div class="status-note">
{note}
</div>
</div>


<div class="summary-card">
<div class="summary-label">
Anomaly Score
</div>

<div class="summary-value">
{score:.4f}
</div>
</div>


<div class="summary-card">
<div class="summary-label">
Threshold
</div>

<div class="summary-value">
{threshold:.4f}
</div>
</div>


<div class="summary-card">
<div class="summary-label">
Difference
</div>

<div class="summary-value">
{difference:+.4f}
</div>
</div>

</div>
""",
                unsafe_allow_html=True,
            )


            # =================================================
            # LOCALIZATION
            # =================================================

            st.markdown(
                """
<div class="section-title">
🎯 Defect Localization
</div>
""",
                unsafe_allow_html=True,
            )


            colored_heatmap = (
                make_heatmap(
                    heatmap
                )
            )

            overlay = (
                make_overlay(
                    display_image,
                    heatmap
                )
            )


            image_col1, image_col2, image_col3 = (
                st.columns(
                    3,
                    gap="small",
                )
            )


            # SMALLER IMAGES FOR DEMO
            with image_col1:

                st.markdown(
                    """
<div class="image-heading">
🖼️ Input
</div>
""",
                    unsafe_allow_html=True,
                )

                st.image(
                    display_image,
                    width=240,
                )


            with image_col2:

                st.markdown(
                    """
<div class="image-heading">
🌡️ Anomaly Heatmap
</div>
""",
                    unsafe_allow_html=True,
                )

                st.image(
                    colored_heatmap,
                    width=240,
                )


            with image_col3:

                st.markdown(
                    """
<div class="image-heading">
🎯 Localization
</div>
""",
                    unsafe_allow_html=True,
                )

                st.image(
                    overlay,
                    width=240,
                )


            with st.expander(
                "Technical details"
            ):

                detail1, detail2 = (
                    st.columns(2)
                )

                with detail1:

                    st.write(
                        "**Model:** "
                        "PatchCore-style"
                    )

                    st.write(
                        "**Backbone:** "
                        "WideResNet50-2"
                    )

                    st.write(
                        "**Features:** "
                        "layer2 + layer3"
                    )

                with detail2:

                    st.write(
                        "**Embedding:** "
                        "128 dimensions"
                    )

                    st.write(
                        "**Memory bank:** "
                        "500 patches"
                    )

                    st.write(
                        "**Calibration:** "
                        f"{config['threshold_method']}"
                    )


    # ========================================================
    # VIDEO MODE
    # ========================================================

    else:

        st.markdown(
            f"""
<div class="section-title">
🎬 {selected_category} Video Inspection
</div>
""",
            unsafe_allow_html=True,
        )


        controls1, controls2 = (
            st.columns(2)
        )

        with controls1:

            sample_every = st.slider(
                "Frame interval",
                min_value=5,
                max_value=30,
                value=10,
                step=5,
            )

        with controls2:

            max_samples = st.slider(
                "Maximum samples",
                min_value=10,
                max_value=120,
                value=40,
                step=10,
            )


        uploaded_video = st.file_uploader(
            "Upload inspection video",
            type=[
                "mp4",
                "avi",
                "mov",
            ],
            key="video_upload",
        )


        if uploaded_video is None:

            st.info(
                "Upload a video to begin inspection."
            )


        else:

            video_bytes = (
                uploaded_video.getvalue()
            )

            video_hash = (
                hashlib.md5(
                    video_bytes
                ).hexdigest()
            )


            video_left, video_right = (
                st.columns(
                    [1.05, 1.55]
                )
            )


            with video_left:

                st.video(
                    video_bytes
                )


            result_key = (
                f"{video_hash}_"
                f"{selected_category}_"
                f"{sample_every}_"
                f"{max_samples}"
            )


            with video_right:

                if st.button(
                    "🚀 Run Video Inspection",
                    width="stretch",
                ):

                    runtime_dir = Path(
                        "runtime"
                    )

                    runtime_dir.mkdir(
                        exist_ok=True
                    )

                    extension = (
                        Path(
                            uploaded_video.name
                        ).suffix
                        or ".mp4"
                    )

                    temporary_path = (
                        runtime_dir
                        /
                        f"{video_hash}"
                        f"{extension}"
                    )

                    temporary_path.write_bytes(
                        video_bytes
                    )


                    progress = st.progress(
                        0
                    )

                    status = st.empty()


                    def update_progress(
                        value,
                    ):

                        progress.progress(
                            float(value)
                        )


                    try:

                        status.info(
                            "Analyzing sampled frames..."
                        )

                        video_result = (
                            inspect_video(
                                video_path=(
                                    temporary_path
                                ),
                                engine=engine,
                                sample_every=(
                                    sample_every
                                ),
                                max_samples=(
                                    max_samples
                                ),
                                progress_callback=(
                                    update_progress
                                ),
                            )
                        )

                        st.session_state[
                            "video_result"
                        ] = video_result

                        st.session_state[
                            "video_result_key"
                        ] = result_key

                        status.success(
                            "Inspection complete"
                        )

                    except Exception as error:

                        status.error(
                            f"Inspection failed: "
                            f"{error}"
                        )


                if (
                    st.session_state.get(
                        "video_result_key"
                    )
                    == result_key
                ):

                    video_result = (
                        st.session_state[
                            "video_result"
                        ]
                    )

                    prediction = (
                        video_result[
                            "overall_prediction"
                        ]
                    )

                    if prediction == "NORMAL":

                        status_class = (
                            "status-normal"
                        )

                        icon = "✅"

                    else:

                        status_class = (
                            "status-anomaly"
                        )

                        icon = "⚠️"


                    st.markdown(
                        f"""
<div class="summary-grid">

<div class="summary-card {status_class}">
<div class="summary-label">
Video Result
</div>

<div class="summary-value">
{icon} {prediction}
</div>
</div>


<div class="summary-card">
<div class="summary-label">
Analyzed
</div>

<div class="summary-value">
{video_result["analyzed_frames"]}
</div>
</div>


<div class="summary-card">
<div class="summary-label">
Anomalous
</div>

<div class="summary-value">
{video_result["anomalous_frames"]}
</div>
</div>


<div class="summary-card">
<div class="summary-label">
Max Score
</div>

<div class="summary-value">
{video_result["max_score"]:.4f}
</div>
</div>

</div>
""",
                        unsafe_allow_html=True,
                    )


            if (
                st.session_state.get(
                    "video_result_key"
                )
                == result_key
            ):

                video_result = (
                    st.session_state[
                        "video_result"
                    ]
                )

                records = (
                    video_result[
                        "records"
                    ]
                )

                best = (
                    video_result[
                        "best_frame"
                    ]
                )


                timeline_col, frame_col = (
                    st.columns(
                        [1.25, 1]
                    )
                )


                with timeline_col:

                    st.markdown(
                        """
<div class="image-heading">
📈 Temporal Anomaly Profile
</div>
""",
                        unsafe_allow_html=True,
                    )

                    times = [
                        item["time_seconds"]
                        for item
                        in records
                    ]

                    scores = [
                        item["score"]
                        for item
                        in records
                    ]


                    fig, ax = plt.subplots(
                        figsize=(7.5, 2.3)
                    )

                    fig.patch.set_alpha(
                        0
                    )

                    ax.set_facecolor(
                        "#0f172a"
                    )

                    ax.plot(
                        times,
                        scores,
                        linewidth=2,
                        color="#38bdf8",
                    )

                    ax.axhline(
                        config[
                            "threshold"
                        ],
                        linestyle="--",
                        linewidth=1.5,
                        color="#f87171",
                    )

                    ax.set_xlabel(
                        "Time (s)",
                        color="#cbd5e1",
                    )

                    ax.set_ylabel(
                        "Score",
                        color="#cbd5e1",
                    )

                    ax.tick_params(
                        colors="#94a3b8",
                        labelsize=8,
                    )

                    for spine in (
                        ax.spines.values()
                    ):

                        spine.set_color(
                            "#334155"
                        )

                    fig.tight_layout()

                    st.pyplot(
                        fig,
                        width="stretch",
                    )

                    plt.close(
                        fig
                    )


                with frame_col:

                    st.markdown(
                        f"""
<div class="image-heading">
🔥 Peak Frame •
{best["time_seconds"]:.2f}s •
Score {best["score"]:.4f}
</div>
""",
                        unsafe_allow_html=True,
                    )


                    best_overlay = (
                        make_overlay(
                            best[
                                "image"
                            ],
                            best[
                                "heatmap"
                            ],
                        )
                    )

                    st.image(
                        best_overlay,
                        width=260,
                    )


# ============================================================
# BENCHMARK TAB
# ============================================================

with benchmark_tab:

    st.markdown(
        """
<div class="section-title">
📊 VisionInspect Development Evaluation
</div>
""",
        unsafe_allow_html=True,
    )

    st.caption(
        "Experimental results across Bottle, Hazelnut "
        "and Tile from the MVTec Anomaly Detection dataset."
    )


    category_results = load_results_csv(
        "results/patchcore_three_category_results.csv"
    )

    comparison_results = load_results_csv(
        "results/bottle_model_comparison.csv"
    )

    ablation_results = load_results_csv(
        "results/bottle_coreset_ablation.csv"
    )


    # ========================================================
    # OVERALL
    # ========================================================

    st.markdown(
        "### 🏆 Overall Performance"
    )


    if category_results is not None:

        image_col = find_column(
            category_results,
            [
                "image_auroc",
                "Image AUROC",
            ],
        )

        pixel_col = find_column(
            category_results,
            [
                "pixel_auroc",
                "Pixel AUROC",
            ],
        )

        f1_col = find_column(
            category_results,
            [
                "f1",
                "F1",
                "f1_score",
            ],
        )


        avg_image = (
            category_results[
                image_col
            ].mean()
            if image_col
            else 1.0
        )

        avg_pixel = (
            category_results[
                pixel_col
            ].mean()
            if pixel_col
            else 0.9696
        )

        avg_f1 = (
            category_results[
                f1_col
            ].mean()
            if f1_col
            else 0.983
        )

    else:

        avg_image = 1.0
        avg_pixel = 0.9696
        avg_f1 = 0.983


    overall1, overall2, overall3 = (
        st.columns(3)
    )


    with overall1:

        st.metric(
            "Average Image AUROC",
            f"{avg_image:.4f}",
        )


    with overall2:

        st.metric(
            "Average Pixel AUROC",
            f"{avg_pixel:.4f}",
        )


    with overall3:

        st.metric(
            "Average F1",
            f"{avg_f1:.3f}",
        )


    st.divider()


    # ========================================================
    # CATEGORY PERFORMANCE
    # ========================================================

    st.markdown(
        "### 📈 Category Performance"
    )


    if category_results is not None:

        category_col = find_column(
            category_results,
            [
                "category",
                "Category",
            ],
        )

        image_col = find_column(
            category_results,
            [
                "image_auroc",
                "Image AUROC",
            ],
        )

        pixel_col = find_column(
            category_results,
            [
                "pixel_auroc",
                "Pixel AUROC",
            ],
        )


        if (
            category_col
            and image_col
            and pixel_col
        ):

            categories = (
                category_results[
                    category_col
                ]
                .astype(str)
                .str.title()
            )

            image_values = (
                category_results[
                    image_col
                ]
            )

            pixel_values = (
                category_results[
                    pixel_col
                ]
            )


            x = np.arange(
                len(categories)
            )

            width = 0.34


            fig, ax = plt.subplots(
                figsize=(9, 3.4)
            )

            fig.patch.set_alpha(
                0
            )

            ax.set_facecolor(
                "#0f172a"
            )


            ax.bar(
                x - width / 2,
                image_values,
                width,
                label="Image AUROC",
                color="#38bdf8",
            )

            ax.bar(
                x + width / 2,
                pixel_values,
                width,
                label="Pixel AUROC",
                color="#a78bfa",
            )


            ax.set_xticks(
                x
            )

            ax.set_xticklabels(
                categories,
                color="#cbd5e1",
            )

            ax.set_ylim(
                0.90,
                1.01,
            )

            ax.set_ylabel(
                "AUROC",
                color="#cbd5e1",
            )

            ax.tick_params(
                colors="#94a3b8"
            )

            ax.grid(
                axis="y",
                alpha=0.12,
            )


            for spine in (
                ax.spines.values()
            ):

                spine.set_color(
                    "#334155"
                )


            legend = ax.legend(
                frameon=False
            )

            for text in (
                legend.get_texts()
            ):

                text.set_color(
                    "#cbd5e1"
                )


            fig.tight_layout()

            st.pyplot(
                fig,
                width="stretch",
            )

            plt.close(
                fig
            )


            category1, category2, category3 = (
                st.columns(3)
            )


            category_cards = [
                (
                    category1,
                    "🍾 Bottle",
                    1.0000,
                    0.9819,
                    0.629979,
                ),
                (
                    category2,
                    "🌰 Hazelnut",
                    1.0000,
                    0.9841,
                    0.774364,
                ),
                (
                    category3,
                    "🧱 Tile",
                    1.0000,
                    0.9428,
                    0.669720,
                ),
            ]


            for (
                column,
                name,
                image_score,
                pixel_score,
                threshold,
            ) in category_cards:

                with column:

                    st.markdown(
                        f"#### {name}"
                    )

                    st.metric(
                        "Image AUROC",
                        f"{image_score:.4f}",
                    )

                    st.metric(
                        "Pixel AUROC",
                        f"{pixel_score:.4f}",
                    )

                    st.caption(
                        f"Threshold: "
                        f"{threshold:.4f}"
                    )


    else:

        st.warning(
            "Could not load the three-category "
            "evaluation CSV."
        )


    st.divider()


    # ========================================================
    # BASELINE VS PATCHCORE
    # ========================================================

    st.markdown(
        "### 🧠 Baseline vs PatchCore"
    )

    st.caption(
        "Bottle comparison between the convolutional "
        "autoencoder baseline and final PatchCore-style model."
    )


    if comparison_results is not None:

        model_col = find_column(
            comparison_results,
            [
                "model",
                "Model",
            ],
        )

        image_col = find_column(
            comparison_results,
            [
                "image_auroc",
                "Image AUROC",
            ],
        )

        pixel_col = find_column(
            comparison_results,
            [
                "pixel_auroc",
                "Pixel AUROC",
            ],
        )


        if (
            model_col
            and image_col
            and pixel_col
        ):

            models = (
                comparison_results[
                    model_col
                ].astype(str)
            )

            x = np.arange(
                len(models)
            )

            width = 0.34


            fig, ax = plt.subplots(
                figsize=(8, 3.3)
            )

            fig.patch.set_alpha(
                0
            )

            ax.set_facecolor(
                "#0f172a"
            )


            ax.bar(
                x - width / 2,
                comparison_results[
                    image_col
                ],
                width,
                label="Image AUROC",
                color="#22d3ee",
            )

            ax.bar(
                x + width / 2,
                comparison_results[
                    pixel_col
                ],
                width,
                label="Pixel AUROC",
                color="#f472b6",
            )


            ax.set_xticks(
                x
            )

            ax.set_xticklabels(
                models,
                color="#cbd5e1",
            )

            ax.set_ylim(
                0.65,
                1.02,
            )

            ax.set_ylabel(
                "AUROC",
                color="#cbd5e1",
            )

            ax.tick_params(
                colors="#94a3b8"
            )

            ax.grid(
                axis="y",
                alpha=0.12,
            )


            for spine in (
                ax.spines.values()
            ):

                spine.set_color(
                    "#334155"
                )


            legend = ax.legend(
                frameon=False
            )

            for text in (
                legend.get_texts()
            ):

                text.set_color(
                    "#cbd5e1"
                )


            fig.tight_layout()

            st.pyplot(
                fig,
                width="stretch",
            )

            plt.close(
                fig
            )


    else:

        st.warning(
            "Could not load Bottle model "
            "comparison CSV."
        )


    st.divider()


    # ========================================================
    # CORESET ABLATION
    # ========================================================

    st.markdown(
        "### 🔬 Coreset Ablation Study"
    )

    st.caption(
        "Impact of PatchCore memory-bank size on "
        "localization performance and computational cost."
    )


    if ablation_results is not None:

        coreset_col = find_column(
            ablation_results,
            [
                "coreset_size",
                "Coreset Size",
            ],
        )

        pixel_col = find_column(
            ablation_results,
            [
                "pixel_auroc",
                "Pixel AUROC",
            ],
        )

        inference_col = find_column(
            ablation_results,
            [
                "inference_ms_per_image",
                "inference_ms",
                "inference_time_ms",
                "inference_time",
            ],
        )

        memory_col = find_column(
            ablation_results,
            [
                "memory_mb",
                "memory_size_mb",
                "memory",
            ],
        )


        if (
            coreset_col
            and pixel_col
        ):

            chart1, chart2 = (
                st.columns(2)
            )


            with chart1:

                st.markdown(
                    "#### Localization Performance"
                )

                fig, ax = plt.subplots(
                    figsize=(6, 3.2)
                )

                fig.patch.set_alpha(
                    0
                )

                ax.set_facecolor(
                    "#0f172a"
                )

                ax.plot(
                    ablation_results[
                        coreset_col
                    ],
                    ablation_results[
                        pixel_col
                    ],
                    marker="o",
                    linewidth=2.3,
                    color="#38bdf8",
                )

                ax.set_xlabel(
                    "Coreset Size",
                    color="#cbd5e1",
                )

                ax.set_ylabel(
                    "Pixel AUROC",
                    color="#cbd5e1",
                )

                ax.tick_params(
                    colors="#94a3b8"
                )

                ax.grid(
                    alpha=0.12,
                )

                for spine in (
                    ax.spines.values()
                ):

                    spine.set_color(
                        "#334155"
                    )

                fig.tight_layout()

                st.pyplot(
                    fig,
                    width="stretch",
                )

                plt.close(
                    fig
                )


            with chart2:

                st.markdown(
                    "#### Efficiency Trade-off"
                )


                if inference_col:

                    fig, ax = plt.subplots(
                        figsize=(6, 3.2)
                    )

                    fig.patch.set_alpha(
                        0
                    )

                    ax.set_facecolor(
                        "#0f172a"
                    )

                    ax.plot(
                        ablation_results[
                            coreset_col
                        ],
                        ablation_results[
                            inference_col
                        ],
                        marker="o",
                        linewidth=2.3,
                        color="#f59e0b",
                    )

                    ax.set_xlabel(
                        "Coreset Size",
                        color="#cbd5e1",
                    )

                    ax.set_ylabel(
                        "Inference ms / image",
                        color="#cbd5e1",
                    )

                    ax.tick_params(
                        colors="#94a3b8"
                    )

                    ax.grid(
                        alpha=0.12,
                    )

                    for spine in (
                        ax.spines.values()
                    ):

                        spine.set_color(
                            "#334155"
                        )

                    fig.tight_layout()

                    st.pyplot(
                        fig,
                        width="stretch",
                    )

                    plt.close(
                        fig
                    )


                elif memory_col:

                    fig, ax = plt.subplots(
                        figsize=(6, 3.2)
                    )

                    fig.patch.set_alpha(
                        0
                    )

                    ax.set_facecolor(
                        "#0f172a"
                    )

                    ax.plot(
                        ablation_results[
                            coreset_col
                        ],
                        ablation_results[
                            memory_col
                        ],
                        marker="o",
                        linewidth=2.3,
                        color="#f59e0b",
                    )

                    ax.set_xlabel(
                        "Coreset Size",
                        color="#cbd5e1",
                    )

                    ax.set_ylabel(
                        "Memory (MB)",
                        color="#cbd5e1",
                    )

                    ax.tick_params(
                        colors="#94a3b8"
                    )

                    ax.grid(
                        alpha=0.12,
                    )

                    for spine in (
                        ax.spines.values()
                    ):

                        spine.set_color(
                            "#334155"
                        )

                    fig.tight_layout()

                    st.pyplot(
                        fig,
                        width="stretch",
                    )

                    plt.close(
                        fig
                    )


                else:

                    st.info(
                        "No efficiency column found "
                        "inside the ablation CSV."
                    )


            st.success(
                "A 500-patch memory bank was selected "
                "as the default accuracy-efficiency "
                "trade-off."
            )


            with st.expander(
                "View raw ablation results"
            ):

                st.dataframe(
                    ablation_results,
                    width="stretch",
                    hide_index=True,
                )


    else:

        st.warning(
            "Could not load coreset "
            "ablation CSV."
        )


    st.divider()

    st.info(
        "Results cover Bottle, Hazelnut and Tile only, "
        "not the complete MVTec benchmark."
    )

    st.warning(
        "Threshold diagnostics were inspected during "
        "development. Results are therefore reported as "
        "development evaluation rather than a completely "
        "untouched final benchmark."
    )


# ============================================================
# ARCHITECTURE TAB
# ============================================================

with architecture_tab:

    st.markdown(
        """
<div class="section-title">
🧠 VisionInspect Pipeline
</div>
""",
        unsafe_allow_html=True,
    )


    pipeline1, pipeline2 = (
        st.columns(2)
    )


    with pipeline1:

        st.markdown(
            """
**1. Normal-only learning**

Normal industrial samples define the reference distribution.

**2. Deep feature extraction**

WideResNet50-2 extracts `layer2` and `layer3` features.

**3. Multi-scale fusion**

Feature maps are aligned and concatenated into
1536-dimensional local representations.

**4. Feature projection**

Embeddings are projected to 128 dimensions.
"""
        )


    with pipeline2:

        st.markdown(
            """
**5. PatchCore memory bank**

500 representative normal embeddings are retained.

**6. Nearest-neighbour scoring**

Test patches are compared with the normal memory bank.

**7. Image-level detection**

The top 1% most anomalous patches produce the image score.

**8. Localization & video**

Patch scores produce heatmaps and the same detector
is applied to sampled video frames.
"""
        )


    st.divider()

    st.markdown(
        "### 🔄 Processing Pipeline"
    )

    st.code(
        """
Input Image
    ↓
Resize + ImageNet Normalization
    ↓
WideResNet50-2
    ↓
layer2 + layer3
    ↓
Multi-scale Feature Fusion
    ↓
1536-D Local Patch Features
    ↓
128-D Projection
    ↓
Nearest Normal Memory Search
    ↓
Patch Anomaly Scores
    ↓
Image Score + Heatmap
    ↓
NORMAL / ANOMALOUS
""",
        language="text",
    )


    st.warning(
        "The detector is category-specific. Large changes "
        "in viewpoint, lighting, background, scale or "
        "product appearance can produce domain-shift "
        "false positives."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="app-footer">
VisionInspect AI • PatchCore • WideResNet50-2 •
Industrial Anomaly Detection
</div>
""",
    unsafe_allow_html=True,
)