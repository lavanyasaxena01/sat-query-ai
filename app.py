import os
import json
import tempfile
from pathlib import Path


# ============================================================
# 1. PROJECT / HUGGING FACE CACHE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

HF_CACHE = PROJECT_ROOT / "hf_cache"

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# ============================================================
# 2. STREAMLIT
# ============================================================

import streamlit as st
from PIL import Image


# IMPORTANT:
# Import controller only AFTER HF cache variables are set.
from agent.controller import SatQueryController


# ============================================================
# 3. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 4. CSS ONLY
#
# NO HTML CONTENT IS USED ANYWHERE ELSE IN THIS APP.
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(0, 190, 255, 0.08),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(80, 70, 220, 0.08),
                transparent 30%
            ),
            #07111f;
        color: #eaf4f8;
    }

    .main .block-container {
        max-width: 1350px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background: #081522;
        border-right: 1px solid rgba(130, 190, 225, 0.12);
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #dceaf0;
    }


    /* ======================================================
       TEXT
       ====================================================== */

    h1, h2, h3, h4 {
        color: #eef7fa !important;
    }

    p {
        color: #b6cad5;
    }

    label {
        color: #b9cdd7 !important;
    }


    /* ======================================================
       INPUTS
       ====================================================== */

    textarea,
    input {
        background-color: #0c1d2c !important;
        color: #edf7fa !important;
    }

    textarea::placeholder,
    input::placeholder {
        color: #6d8799 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #0c1d2c !important;
        color: #edf7fa !important;
        border-color: rgba(120, 190, 225, 0.18) !important;
    }

    div[data-baseweb="select"] span {
        color: #edf7fa !important;
    }


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {
        background: rgba(9, 24, 38, 0.72);
        border: 1px dashed rgba(36, 199, 255, 0.30);
        border-radius: 14px;
        padding: 0.5rem;
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {
        min-height: 44px;
        border-radius: 10px;
        font-weight: 750;
        color: #eaf7fb;
        background: #0d2131;
        border: 1px solid rgba(36, 199, 255, 0.22);
    }

    .stButton > button:hover {
        border-color: #24c7ff;
        color: #ffffff;
    }

    .stButton > button[kind="primary"] {
        background:
            linear-gradient(
                135deg,
                #087fa8,
                #155ea2
            );
        color: white;
        border: none;
    }


    /* ======================================================
       EXPANDERS
       ====================================================== */

    [data-testid="stExpander"] {
        background: rgba(10, 27, 42, 0.72);
        border: 1px solid rgba(120, 185, 220, 0.13);
        border-radius: 14px;
    }


    /* ======================================================
       METRIC COMPONENTS
       ====================================================== */

    [data-testid="stMetric"] {
        background: rgba(11, 29, 44, 0.85);
        border: 1px solid rgba(120, 185, 220, 0.13);
        border-radius: 14px;
        padding: 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: #7895a8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #eef8fb !important;
    }


    /* ======================================================
       INFO / SUCCESS / WARNING
       ====================================================== */

    [data-testid="stAlert"] {
        border-radius: 12px;
    }


    /* ======================================================
       DIVIDER
       ====================================================== */

    hr {
        border-color: rgba(120, 180, 220, 0.10);
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer-text {
        color: #526d80;
        text-align: center;
        font-size: 0.72rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. SESSION STATE
# ============================================================

if "controller" not in st.session_state:
    st.session_state.controller = SatQueryController()

if "result" not in st.session_state:
    st.session_state.result = None


# ============================================================
# 6. HELPERS
# ============================================================

def save_uploaded_file(uploaded_file):

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temp.write(
        uploaded_file.getbuffer()
    )

    temp.close()

    return temp.name


def json_safe(data):

    return json.dumps(
        data,
        indent=2,
        default=str,
    )


def format_confidence(value):

    if value is None:
        return "—"

    try:
        return f"{float(value) * 100:.2f}%"

    except Exception:
        return str(value)


def get_evidence(
    result,
    evidence_type,
):

    return [
        item
        for item in result.get(
            "evidence",
            [],
        )
        if item.get("type") == evidence_type
    ]


def display_evidence_image(
    evidence_item,
):

    if not evidence_item:
        return

    path = evidence_item.get("path")

    if not path:
        return

    path = Path(path)

    if not path.exists():

        st.warning(
            f"Evidence file not found: {path}"
        )

        return

    try:

        image = Image.open(path)

        st.image(
            image,
            use_container_width=True,
        )

    except Exception as error:

        st.warning(
            f"Could not display image: {error}"
        )


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛰️ SatQuery AI")

    st.caption(
        "Agentic Remote-Sensing Intelligence"
    )

    st.divider()

    st.subheader(
        "Analysis Configuration"
    )

    modality_mode = st.selectbox(
        "Input modality",
        [
            "Auto Detect",
            "Optical",
            "SAR",
            "Optical + SAR",
        ],
    )

    st.divider()

    st.subheader(
        "Model Ecosystem"
    )

    st.markdown(
        """
        **👁️ VQA**

        `Qwen2.5-VL + SatQuery LoRA`

        **📝 Captioning**

        `Qwen2.5-VL + SatQuery LoRA`

        **🔄 Change Detection**

        `Bi-temporal Change Detection`

        **🔍 Change-VQA**

        `Qwen2.5-VL + SatQuery LoRA`

        **🛰️ Optical + SAR**

        `Cross-Modal Fusion Network`
        """
    )

    st.divider()

    st.caption(
        "5 specialist models available"
    )

    if st.button(
        "Clear Current Analysis",
        use_container_width=True,
    ):

        st.session_state.result = None

        st.rerun()


# ============================================================
# 8. HERO
# ============================================================

st.caption(
    "AGENTIC EARTH OBSERVATION"
)

st.title(
    "SATQUERY AI"
)

st.markdown(
    """
    **Understand, compare and interrogate satellite imagery**
    
    using an intelligent multi-model remote-sensing analysis
    pipeline.
    """
)

st.success(
    "● SYSTEM READY"
)

st.divider()


# ============================================================
# 9. INPUT
# ============================================================

st.header(
    "New Satellite Analysis"
)

st.caption(
    "Upload one image for visual intelligence or two images "
    "for temporal and cross-modal analysis."
)


uploaded_files = st.file_uploader(
    "Upload satellite imagery",
    type=[
        "tif",
        "tiff",
        "png",
        "jpg",
        "jpeg",
    ],
    accept_multiple_files=True,
    help=(
        "1 image → VQA / Captioning. "
        "2 images → Change Detection / Change-VQA / Optical + SAR."
    ),
)


# ============================================================
# 10. IMAGE PREVIEW
# ============================================================

if uploaded_files:

    st.subheader(
        "Input Imagery"
    )

    columns = st.columns(
        min(
            len(uploaded_files),
            2,
        )
    )

    for index, uploaded_file in enumerate(
        uploaded_files
    ):

        with columns[
            index % len(columns)
        ]:

            st.caption(
                f"IMAGE {index + 1} · "
                f"{uploaded_file.name}"
            )

            try:

                preview = Image.open(
                    uploaded_file
                )

                st.image(
                    preview,
                    use_container_width=True,
                )

            except Exception:

                st.info(
                    "Raster uploaded. "
                    "Preview is unavailable."
                )


# ============================================================
# 11. QUERY
# ============================================================

st.subheader(
    "Natural-Language Query"
)

query = st.text_area(
    "Ask SatQuery AI",
    placeholder=(
        "What type of land cover is visible?\n\n"
        "Describe this satellite scene.\n\n"
        "What changed between these two images?\n\n"
        "Did the urban area increase?\n\n"
        "Analyze these optical and SAR images together."
    ),
    height=140,
    label_visibility="collapsed",
)


# ============================================================
# 12. MODALITY
# ============================================================

declared_modalities = None

if modality_mode == "Optical":

    declared_modalities = [
        "optical"
    ]

elif modality_mode == "SAR":

    declared_modalities = [
        "sar"
    ]

elif modality_mode == "Optical + SAR":

    declared_modalities = [
        "optical",
        "sar",
    ]


# ============================================================
# 13. ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "✦  ANALYZE IMAGERY",
    type="primary",
    use_container_width=True,
)


if analyze:

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not uploaded_files:

        st.error(
            "Please upload at least one image."
        )

        st.stop()

    if not query.strip():

        st.error(
            "Please enter a natural-language query."
        )

        st.stop()

    if len(uploaded_files) > 2:

        st.error(
            "A maximum of two images is currently supported."
        )

        st.stop()

    if (
        modality_mode == "Optical + SAR"
        and len(uploaded_files) != 2
    ):

        st.error(
            "Optical + SAR requires exactly two images."
        )

        st.stop()


    # --------------------------------------------------------
    # SAVE UPLOADS
    # --------------------------------------------------------

    image_paths = []

    for uploaded_file in uploaded_files:

        image_paths.append(
            save_uploaded_file(
                uploaded_file
            )
        )


    # --------------------------------------------------------
    # CONTROLLER
    # --------------------------------------------------------

    with st.spinner(
        "SatQuery AI is routing and analyzing the imagery..."
    ):

        try:

            result = (
                st.session_state.controller.analyze(
                    query=query,
                    images=image_paths,
                    modalities=declared_modalities,
                )
            )

            st.session_state.result = result

        except Exception as error:

            st.session_state.result = {
                "success": False,
                "error": str(error),
            }


# ============================================================
# 14. RESULTS
# ============================================================

result = st.session_state.result


if result:

    st.divider()

    if not result.get(
        "success",
        False,
    ):

        st.error(
            result.get(
                "error",
                "Analysis failed.",
            )
        )

    else:

        # ====================================================
        # RESULT HEADER
        # ====================================================

        st.caption(
            "ANALYSIS COMPLETE"
        )

        task = result.get(
            "task",
            "unknown",
        )

        model = result.get(
            "model",
            "Unknown",
        )

        routing_confidence = result.get(
            "routing_confidence"
        )

        confidence = result.get(
            "confidence"
        )

        # ====================================================
        # TOP METRICS
        # ====================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Selected Task",
                task.replace(
                    "_",
                    " ",
                ).title(),
            )

        with col2:

            st.metric(
                "Model",
                model,
            )

        with col3:

            st.metric(
                "Routing Confidence",
                format_confidence(
                    routing_confidence
                ),
            )

        with col4:

            st.metric(
                "Model Confidence",
                format_confidence(
                    confidence
                ),
            )


        # ====================================================
        # ANSWER
        # ====================================================

        st.subheader(
            "SatQuery AI Result"
        )

        answer = result.get(
            "answer",
            "No answer generated.",
        )

        st.info(
            answer
        )


        # ====================================================
        # AGENT ROUTING
        # ====================================================

        with st.expander(
            "🧠 Agent Decision & Routing",
            expanded=True,
        ):

            routing_reason = result.get(
                "routing_reason",
                "No routing reason available.",
            )

            st.write(
                "**Selected task:**",
                task,
            )

            st.write(
                "**Routing reason:**",
                routing_reason,
            )

            scores = result.get(
                "candidate_scores",
                {},
            )

            if scores:

                st.markdown(
                    "#### Candidate Scores"
                )

                for candidate, score in scores.items():

                    st.write(
                        candidate.replace(
                            "_",
                            " ",
                        ).title()
                    )

                    try:

                        score_value = float(
                            score
                        )

                        st.progress(
                            min(
                                score_value / 10,
                                1.0,
                            )
                        )

                    except Exception:

                        st.write(
                            f"Score: {score}"
                        )


        # ====================================================
        # VISUAL EVIDENCE
        # ====================================================

        st.header(
            "Visual Evidence"
        )


        # ----------------------------------------------------
        # CHANGE DETECTION / CHANGE VQA
        # ----------------------------------------------------

        if task in [
            "change_detection",
            "change_vqa",
        ]:

            input_images = get_evidence(
                result,
                "input_image",
            )

            change_maps = get_evidence(
                result,
                "change_map",
            )

            cols = st.columns(3)

            if len(input_images) >= 1:

                with cols[0]:

                    st.subheader(
                        "Before"
                    )

                    display_evidence_image(
                        input_images[0]
                    )

            if len(input_images) >= 2:

                with cols[1]:

                    st.subheader(
                        "After"
                    )

                    display_evidence_image(
                        input_images[1]
                    )

            if change_maps:

                with cols[2]:

                    st.subheader(
                        "Change Map"
                    )

                    display_evidence_image(
                        change_maps[0]
                    )


        # ----------------------------------------------------
        # OPTICAL + SAR
        # ----------------------------------------------------

        elif task == "optical_sar":

            input_images = get_evidence(
                result,
                "input_image",
            )

            fusion_images = get_evidence(
                result,
                "optical_sar_fusion",
            )

            cols = st.columns(3)

            if len(input_images) >= 1:

                with cols[0]:

                    st.subheader(
                        "Optical"
                    )

                    display_evidence_image(
                        input_images[0]
                    )

            if len(input_images) >= 2:

                with cols[1]:

                    st.subheader(
                        "SAR"
                    )

                    display_evidence_image(
                        input_images[1]
                    )

            if fusion_images:

                with cols[2]:

                    st.subheader(
                        "Fusion Evidence"
                    )

                    display_evidence_image(
                        fusion_images[0]
                    )


        # ----------------------------------------------------
        # VQA / CAPTIONING
        # ----------------------------------------------------

        else:

            input_images = get_evidence(
                result,
                "input_image",
            )

            if input_images:

                for item in input_images:

                    display_evidence_image(
                        item
                    )


        # ====================================================
        # CHANGE DETECTION DETAILS
        # ====================================================

        if task == "change_detection":

            st.header(
                "Change Analysis"
            )

            model_result = result.get(
                "model_result",
                {}
            )

            if not model_result:

                # Controller versions may use these fields
                # directly instead.
                changed_pixels = result.get(
                    "changed_pixels"
                )

                total_pixels = result.get(
                    "total_pixels"
                )

                change_percentage = result.get(
                    "change_percentage"
                )

                image_size = result.get(
                    "image_size"
                )

            else:

                changed_pixels = model_result.get(
                    "changed_pixels"
                )

                total_pixels = model_result.get(
                    "total_pixels"
                )

                change_percentage = model_result.get(
                    "change_percentage"
                )

                image_size = model_result.get(
                    "image_size"
                )


            c1, c2, c3 = st.columns(3)

            with c1:

                if change_percentage is not None:

                    st.metric(
                        "Changed Area",
                        f"{change_percentage:.2f}%",
                    )

                else:

                    st.metric(
                        "Changed Area",
                        "—",
                    )

            with c2:

                if changed_pixels is not None:

                    st.metric(
                        "Changed Pixels",
                        f"{changed_pixels:,}",
                    )

                else:

                    st.metric(
                        "Changed Pixels",
                        "—",
                    )

            with c3:

                if image_size:

                    if isinstance(
                        image_size,
                        dict,
                    ):

                        width = image_size.get(
                            "width"
                        )

                        height = image_size.get(
                            "height"
                        )

                        st.metric(
                            "Analysis Size",
                            f"{width} × {height}",
                        )

                    else:

                        st.metric(
                            "Analysis Size",
                            str(image_size),
                        )

                else:

                    st.metric(
                        "Analysis Size",
                        "—",
                    )


            st.caption(
                "Confidence type: "
                + str(
                    result.get(
                        "confidence_type",
                        "heuristic diagnostic",
                    )
                )
            )


        # ====================================================
        # OPTICAL + SAR DETAILS
        # ====================================================

        if task == "optical_sar":

            st.header(
                "Optical + SAR Fusion"
            )

            model_result = result.get(
                "model_result",
                {}
            )

            if not model_result:

                model_result = result


            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Fusion Type",
                    "Cross-Modal",
                )

            with c2:

                shape = model_result.get(
                    "fused_features_shape"
                )

                if shape:

                    st.metric(
                        "Feature Shape",
                        " × ".join(
                            map(
                                str,
                                shape,
                            )
                        ),
                    )

                else:

                    st.metric(
                        "Feature Shape",
                        "Available",
                    )

            with c3:

                st.metric(
                    "Confidence",
                    format_confidence(
                        result.get(
                            "confidence"
                        )
                    ),
                )


            st.caption(
                "Confidence type: "
                + str(
                    result.get(
                        "confidence_type",
                        "heuristic diagnostic",
                    )
                )
            )


        # ====================================================
        # MODALITY
        # ====================================================

        with st.expander(
            "🛰️ Modality Detection",
            expanded=False,
        ):

            modalities = result.get(
                "modalities",
                [],
            )

            if modalities:

                for item in modalities:

                    st.markdown(
                        f"""
                        **{str(
                            item.get(
                                "modality",
                                "unknown",
                            )
                        ).upper()}**

                        Confidence:
                        {format_confidence(
                            item.get(
                                "confidence"
                            )
                        )}

                        Source:
                        `{item.get(
                            "source",
                            "automatic",
                        )}`

                        Reason:
                        {item.get(
                            "reason",
                            "",
                        )}
                        """
                    )

                    st.divider()

            else:

                st.info(
                    "No modality information available."
                )


        # ====================================================
        # COMPATIBILITY
        # ====================================================

        with st.expander(
            "✓ Compatibility Check",
            expanded=False,
        ):

            compatibility = result.get(
                "compatibility",
                {},
            )

            if compatibility:

                st.success(
                    compatibility.get(
                        "message",
                        "Compatibility checks passed.",
                    )
                )

                st.write(
                    "**Dimension status:**",
                    compatibility.get(
                        "dimension_status",
                        "unknown",
                    ),
                )

                st.write(
                    "**CRS status:**",
                    compatibility.get(
                        "crs_status",
                        "unknown",
                    ),
                )

            else:

                st.info(
                    "Compatibility information unavailable."
                )


        # ====================================================
        # EXECUTION TRACE
        # ====================================================

        with st.expander(
            "⚙️ Agent Execution Trace",
            expanded=False,
        ):

            trace = result.get(
                "trace",
                [],
            )

            if trace:

                for index, item in enumerate(
                    trace,
                    start=1,
                ):

                    status = item.get(
                        "status",
                        "unknown",
                    )

                    step = item.get(
                        "step",
                        "Unknown step",
                    )

                    details = item.get(
                        "details",
                        "",
                    )

                    if status == "completed":

                        st.success(
                            f"{index}. {step}\n\n"
                            f"{details}"
                        )

                    else:

                        st.warning(
                            f"{index}. {step}\n\n"
                            f"{details}"
                        )

            else:

                st.info(
                    "No execution trace available."
                )


        # ====================================================
        # RAW RESPONSE
        # ====================================================

        with st.expander(
            "🔬 Raw Controller Response",
            expanded=False,
        ):

            st.json(
                result
            )


        # ====================================================
        # DOWNLOAD REPORT
        # ====================================================

        st.header(
            "Analysis Report"
        )

        report_data = json_safe(
            result
        )

        st.download_button(
            label="⬇ Download Analysis Report",
            data=report_data,
            file_name=(
                "satquery_analysis_report.json"
            ),
            mime="application/json",
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SATQUERY AI · Agentic Remote-Sensing Intelligence"
)