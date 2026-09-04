import os
import tempfile
import streamlit as st
from agent.controller import SatQueryController

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-top: 0;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    }

    .metric-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
    }

    .trace-completed {
        color: green;
    }

    .trace-failed {
        color: red;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛰️ SatQuery AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Agentic Vision-Language Intelligence for Remote Sensing
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Upload satellite imagery and ask questions in natural language. "
    "SatQuery AI automatically selects the appropriate analysis workflow."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Analysis Settings")

    st.markdown(
        """
        **Supported formats**

        - GeoTIFF
        - TIFF
        - PNG
        - JPEG

        **Supported analysis**

        - Visual Question Answering
        - Image Captioning
        - Bi-temporal Change Detection
        - Change-VQA
        - Optical + SAR Analysis
        """
    )

    st.divider()

    st.info(
        "For Optical + SAR analysis, upload one optical "
        "image and one SAR image."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📡 Satellite Imagery")

uploaded_files = st.file_uploader(
    "Upload satellite image(s)",
    type=[
        "tif",
        "tiff",
        "png",
        "jpg",
        "jpeg",
    ],
    accept_multiple_files=True,
)


# ============================================================
# IMAGE PREVIEW
# ============================================================

if uploaded_files:

    st.write("#### Uploaded Images")

    columns = st.columns(
        min(len(uploaded_files), 3)
    )

    for index, uploaded_file in enumerate(
        uploaded_files
    ):

        with columns[index % len(columns)]:

            st.write(
                f"**Image {index + 1}**"
            )

            st.caption(
                uploaded_file.name
            )

            # Streamlit can preview common image formats.
            # TIFF/GeoTIFF preview may not always work.
            try:

                if uploaded_file.name.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):

                    st.image(
                        uploaded_file,
                        use_container_width=True,
                    )

                else:

                    st.info(
                        "GeoTIFF/TIFF uploaded successfully."
                    )

            except Exception:

                st.info(
                    "Preview unavailable for this format."
                )


st.divider()


# ============================================================
# MODALITY SELECTION
# ============================================================

st.subheader("🛰️ Image Modality")

if len(uploaded_files) > 0:

    modality_options = [
        "Auto Detect",
        "Optical",
        "SAR",
    ]

    selected_modalities = []

    for index, uploaded_file in enumerate(
        uploaded_files
    ):

        modality = st.selectbox(
            f"Image {index + 1}: {uploaded_file.name}",
            modality_options,
            key=f"modality_{index}",
        )

        selected_modalities.append(
            modality.lower().replace(
                "auto detect",
                "auto",
            )
        )

else:

    selected_modalities = []


st.divider()


# ============================================================
# QUERY
# ============================================================

st.subheader("💬 Natural Language Query")

query = st.text_area(
    "Ask SatQuery AI",
    placeholder=(
        "Examples:\n"
        "• What is visible in this satellite image?\n"
        "• Describe this satellite scene.\n"
        "• What changed between these two images?\n"
        "• Compare the two images and describe the changes.\n"
        "• Analyze the optical and SAR imagery together."
    ),
    height=120,
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🚀 Analyze with SatQuery AI",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if not uploaded_files:

        st.error(
            "Please upload at least one satellite image."
        )

        st.stop()

    if not query.strip():

        st.error(
            "Please enter a natural-language query."
        )

        st.stop()

    if len(uploaded_files) > 2:

        st.error(
            "SatQuery AI currently supports a maximum "
            "of two images per analysis."
        )

        st.stop()

    # --------------------------------------------------------
    # SAVE UPLOADED FILES
    # --------------------------------------------------------

    image_paths = []

    try:

        for uploaded_file in uploaded_files:

            suffix = os.path.splitext(
                uploaded_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                image_paths.append(
                    temp_file.name
                )

        # ----------------------------------------------------
        # CONVERT MODALITY DECLARATIONS
        # ----------------------------------------------------

        declared_modalities = []

        for modality in selected_modalities:

            if modality in [
                "optical",
                "sar",
            ]:

                declared_modalities.append(
                    modality
                )

            else:

                declared_modalities.append(
                    None
                )

        # If everything is Auto Detect, don't explicitly
        # declare modalities.
        if all(
            modality is None
            for modality in declared_modalities
        ):

            declared_modalities = None

        # ----------------------------------------------------
        # RUN CONTROLLER
        # ----------------------------------------------------

        controller = SatQueryController()

        with st.spinner(
            "SatQuery AI is analyzing the imagery..."
        ):

            result = controller.analyze(
                query=query,
                images=image_paths,
                modalities=declared_modalities,
            )

        # ====================================================
        # RESULT
        # ====================================================

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

            # Show trace even on failure.
            trace = result.get(
                "trace",
                [],
            )

            if trace:

                with st.expander(
                    "🔍 Execution Trace",
                    expanded=True,
                ):

                    for step in trace:

                        status = step.get(
                            "status",
                            "unknown",
                        )

                        icon = (
                            "✅"
                            if status == "completed"
                            else "❌"
                        )

                        st.write(
                            f"{icon} **{step.get('step')}** — "
                            f"{step.get('details', '')}"
                        )

            st.stop()

        # ====================================================
        # TASK / AGENT DECISION
        # ====================================================

        st.subheader(
            "🤖 Agent Decision"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Selected Task",
                result.get(
                    "task_description",
                    result.get("task", "Unknown"),
                ),
            )

        with col2:

            routing_confidence = result.get(
                "routing_confidence"
            )

            if routing_confidence is not None:

                st.metric(
                    "Routing Confidence",
                    f"{routing_confidence:.2f}",
                )

            else:

                st.metric(
                    "Routing Confidence",
                    "N/A",
                )

        with col3:

            st.metric(
                "Selected Model",
                result.get(
                    "model",
                    "Unknown",
                ),
            )

        routing_reason = result.get(
            "routing_reason"
        )

        if routing_reason:

            st.info(
                f"**Agent reasoning:** {routing_reason}"
            )

        # ====================================================
        # ANSWER
        # ====================================================

        st.subheader(
            "🧠 Analysis Result"
        )

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True,
        )

        st.write(
            result.get(
                "answer",
                "No answer generated.",
            )
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        # ====================================================
        # CONFIDENCE
        # ====================================================

        st.subheader(
            "📊 Confidence"
        )

        confidence = result.get(
            "confidence"
        )

        confidence_type = result.get(
            "confidence_type",
            "unavailable",
        )

        c1, c2 = st.columns(2)

        with c1:

            if confidence is not None:

                st.metric(
                    "Model Confidence",
                    f"{confidence:.4f}",
                )

            else:

                st.metric(
                    "Model Confidence",
                    "N/A",
                )

        with c2:

            st.metric(
                "Confidence Type",
                confidence_type,
            )

        # ====================================================
        # MODALITIES
        # ====================================================

        st.subheader(
            "🛰️ Detected Modalities"
        )

        modalities_result = result.get(
            "modalities",
            [],
        )

        if modalities_result:

            modality_columns = st.columns(
                len(modalities_result)
            )

            for index, modality_info in enumerate(
                modalities_result
            ):

                with modality_columns[index]:

                    st.metric(
                        f"Image {index + 1}",
                        modality_info.get(
                            "modality",
                            "unknown",
                        ),
                    )

                    modality_confidence = (
                        modality_info.get(
                            "confidence"
                        )
                    )

                    if modality_confidence is not None:

                        st.caption(
                            f"Detection confidence: "
                            f"{modality_confidence:.2f}"
                        )

        # ====================================================
        # VISUAL EVIDENCE
        # ====================================================

        st.subheader(
            "🖼️ Visual Evidence"
        )

        evidence_items = result.get(
            "evidence",
            [],
        )

        if evidence_items:

            for evidence_item in evidence_items:

                evidence_type = evidence_item.get(
                    "type",
                    "unknown",
                )

                evidence_path = evidence_item.get(
                    "path"
                )

                description = evidence_item.get(
                    "description",
                    "",
                )

                if evidence_type == "input_image":

                    st.write(
                        f"**Input Image** — {description}"
                    )

                    if os.path.exists(
                        evidence_path
                    ):

                        try:

                            st.image(
                                evidence_path,
                                caption=description,
                                use_container_width=True,
                            )

                        except Exception:

                            st.caption(
                                evidence_path
                            )

                elif evidence_type == "change_map":

                    st.write(
                        "**Change Detection Map**"
                    )

                    if os.path.exists(
                        evidence_path
                    ):

                        st.image(
                            evidence_path,
                            caption=description,
                            use_container_width=True,
                        )

                elif evidence_type == "optical_sar_fusion":

                    st.write(
                        "**Optical-SAR Fusion Output**"
                    )

                    if os.path.exists(
                        evidence_path
                    ):

                        st.image(
                            evidence_path,
                            caption=description,
                            use_container_width=True,
                        )

        else:

            st.info(
                "No visual evidence was generated."
            )

        # ====================================================
        # EXECUTION TRACE
        # ====================================================

        st.subheader(
            "🔎 Execution Trace"
        )

        trace = result.get(
            "trace",
            [],
        )

        with st.expander(
            "View complete agent execution trace",
            expanded=False,
        ):

            for step in trace:

                status = step.get(
                    "status",
                    "unknown",
                )

                icon = (
                    "✅"
                    if status == "completed"
                    else "❌"
                )

                st.write(
                    f"{icon} **{step.get('step')}**"
                )

                if step.get(
                    "details"
                ):

                    st.caption(
                        step.get(
                            "details"
                        )
                    )

        # ====================================================
        # CANDIDATE SCORES
        # ====================================================

        candidate_scores = result.get(
            "candidate_scores",
            {},
        )

        if candidate_scores:

            with st.expander(
                "🧩 Agent Candidate Scores"
            ):

                for candidate, score in (
                    candidate_scores.items()
                ):

                    st.write(
                        f"**{candidate}**: {score}"
                    )

        # ====================================================
        # COMPATIBILITY
        # ====================================================

        compatibility = result.get(
            "compatibility",
            {},
        )

        if compatibility:

            with st.expander(
                "🔧 Input Compatibility"
            ):

                st.json(
                    compatibility
                )

        # ====================================================
        # DOWNLOAD REPORT
        # ====================================================

        st.subheader(
            "📄 Analysis Report"
        )

        report_lines = []

        report_lines.append(
            "SATQUERY AI ANALYSIS REPORT"
        )

        report_lines.append(
            "=" * 50
        )

        report_lines.append(
            f"Query: {query}"
        )

        report_lines.append(
            f"Task: {result.get('task_description')}"
        )

        report_lines.append(
            f"Model: {result.get('model')}"
        )

        report_lines.append(
            f"Routing Confidence: "
            f"{result.get('routing_confidence')}"
        )

        report_lines.append(
            f"Model Confidence: "
            f"{result.get('confidence')}"
        )

        report_lines.append(
            f"Confidence Type: "
            f"{result.get('confidence_type')}"
        )

        report_lines.append(
            ""
        )

        report_lines.append(
            "ANSWER"
        )

        report_lines.append(
            "-" * 50
        )

        report_lines.append(
            str(
                result.get(
                    "answer",
                    "",
                )
            )
        )

        report_lines.append(
            ""
        )

        report_lines.append(
            "MODALITIES"
        )

        report_lines.append(
            "-" * 50
        )

        for modality_info in modalities_result:

            report_lines.append(
                str(modality_info)
            )

        report_lines.append(
            ""
        )

        report_lines.append(
            "EVIDENCE"
        )

        report_lines.append(
            "-" * 50
        )

        for evidence_item in evidence_items:

            report_lines.append(
                str(evidence_item)
            )

        report_lines.append(
            ""
        )

        report_lines.append(
            "EXECUTION TRACE"
        )

        report_lines.append(
            "-" * 50
        )

        for step in trace:

            report_lines.append(
                f"[{step.get('status', '').upper()}] "
                f"{step.get('step', '')}: "
                f"{step.get('details', '')}"
            )

        report = "\n".join(
            report_lines
        )

        st.download_button(
            label="⬇️ Download Analysis Report",
            data=report,
            file_name="satquery_ai_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    finally:

        # ----------------------------------------------------
        # CLEAN TEMPORARY FILES
        # ----------------------------------------------------

        for path in image_paths:

            try:

                if os.path.exists(path):
                    os.remove(path)

            except Exception:

                pass