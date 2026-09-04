from agent.router import (
    identify_task,
    route_query,
    get_task_description,
)

from agent.registry import (
    get_model_info,
    get_model,
)

from agent.compatibility import validate_inputs

from agent.modality import (
    detect_modalities,
    check_optical_sar_pair,
)

from agent.evidence import (
    create_multi_image_evidence,
    create_change_map_evidence,
    create_optical_sar_evidence,
)


class SatQueryController:
    """
    Main SATQUERY AI agent controller.

    Workflow:

        Input
          ↓
        Basic Validation
          ↓
        Initial Task Identification
          ↓
        Compatibility Check
          ↓
        Modality Detection
          ↓
        Agentic Task Routing
          ↓
        Model Selection
          ↓
        Model Loading
          ↓
        Model Execution
          ↓
        Evidence Generation
          ↓
        Result + Confidence + Trace
    """

    def __init__(self):
        self.execution_trace = []

    # ========================================================
    # TRACE
    # ========================================================

    def add_trace(
        self,
        step: str,
        status: str,
        details: str = "",
    ):
        self.execution_trace.append(
            {
                "step": step,
                "status": status,
                "details": details,
            }
        )

    def reset_trace(self):
        self.execution_trace = []

    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze(
        self,
        query: str,
        images: list,
        modalities: list[str] | None = None,
    ) -> dict:

        self.reset_trace()

        # ====================================================
        # STEP 1: INPUT
        # ====================================================

        self.add_trace(
            step="Input",
            status="completed",
            details=f"Received {len(images)} image(s).",
        )

        # ====================================================
        # STEP 2: BASIC VALIDATION
        # ====================================================

        if not images:
            self.add_trace(
                step="Input Validation",
                status="failed",
                details="No images provided.",
            )

            return {
                "success": False,
                "error": "No images provided.",
                "trace": self.execution_trace,
            }

        if not query or not query.strip():
            self.add_trace(
                step="Input Validation",
                status="failed",
                details="No query provided.",
            )

            return {
                "success": False,
                "error": "No query provided.",
                "trace": self.execution_trace,
            }

        self.add_trace(
            step="Input Validation",
            status="completed",
            details="Query and basic image inputs are valid.",
        )

        # ====================================================
        # STEP 3: INITIAL TASK IDENTIFICATION
        # ====================================================

        initial_task = identify_task(
            query=query,
            number_of_images=len(images),
        )

        self.add_trace(
            step="Initial Task Identification",
            status="completed",
            details=(
                f"Initial task hypothesis: "
                f"{get_task_description(initial_task)}"
            ),
        )

        # ====================================================
        # STEP 4: COMPATIBILITY CHECK
        # ====================================================

        try:
            compatibility = validate_inputs(
                images=images,
                query=query,
                task=initial_task,
            )

        except Exception as error:
            self.add_trace(
                step="Compatibility Check",
                status="failed",
                details=str(error),
            )

            return {
                "success": False,
                "task": initial_task,
                "error": str(error),
                "trace": self.execution_trace,
            }

        if not compatibility.get("valid", False):

            error_message = compatibility.get(
                "error",
                "Input compatibility check failed.",
            )

            self.add_trace(
                step="Compatibility Check",
                status="failed",
                details=error_message,
            )

            return {
                "success": False,
                "task": initial_task,
                "error": error_message,
                "compatibility": compatibility,
                "trace": self.execution_trace,
            }

        self.add_trace(
            step="Compatibility Check",
            status="completed",
            details=(
                compatibility.get(
                    "message",
                    "Input compatibility check passed.",
                )
            ),
        )

        # ====================================================
        # STEP 5: MODALITY DETECTION
        # ====================================================

        try:
            modality_results = detect_modalities(
                images=images,
                declared_modalities=modalities,
            )

        except Exception as error:

            self.add_trace(
                step="Modality Detection",
                status="failed",
                details=str(error),
            )

            return {
                "success": False,
                "task": initial_task,
                "error": str(error),
                "trace": self.execution_trace,
            }

        modality_names = [
            item.get("modality", "unknown")
            for item in modality_results
        ]

        self.add_trace(
            step="Modality Detection",
            status="completed",
            details=(
                "Detected modalities: "
                + ", ".join(modality_names)
            ),
        )

        # ====================================================
        # STEP 6: AGENTIC TASK ROUTING
        # ====================================================

        routing_decision = route_query(
            query=query,
            number_of_images=len(images),
            modalities=modality_names,
        )

        task = routing_decision["task"]

        task_description = get_task_description(task)

        routing_confidence = routing_decision.get(
            "confidence",
            0.0,
        )

        routing_reason = routing_decision.get(
            "reason",
            "No routing reason provided.",
        )

        candidate_scores = routing_decision.get(
            "candidates",
            {},
        )

        self.add_trace(
            step="Agentic Task Routing",
            status="completed",
            details=(
                f"Selected: {task_description}. "
                f"Routing confidence: "
                f"{routing_confidence}. "
                f"Reason: {routing_reason}"
            ),
        )

        # ====================================================
        # STEP 7: OPTICAL + SAR COMPATIBILITY
        # ====================================================

        if task == "optical_sar":

            pair_check = check_optical_sar_pair(
                images=images,
                declared_modalities=modalities,
            )

            if not pair_check.get(
                "is_optical_sar",
                False,
            ):

                message = pair_check.get(
                    "message",
                    "Inputs could not be verified as "
                    "an Optical + SAR pair.",
                )

                self.add_trace(
                    step="Modality Compatibility",
                    status="failed",
                    details=message,
                )

                return {
                    "success": False,
                    "task": task,
                    "task_description": task_description,
                    "error": message,
                    "modalities": modality_results,
                    "compatibility": compatibility,
                    "routing": routing_decision,
                    "trace": self.execution_trace,
                }

            self.add_trace(
                step="Modality Compatibility",
                status="completed",
                details=(
                    "Verified one optical image and "
                    "one SAR image."
                ),
            )

        # ====================================================
        # STEP 8: MODEL SELECTION
        # ====================================================

        try:
            model_info = get_model_info(task)

        except ValueError as error:

            self.add_trace(
                step="Model Selection",
                status="failed",
                details=str(error),
            )

            return {
                "success": False,
                "task": task,
                "task_description": task_description,
                "routing": routing_decision,
                "error": str(error),
                "trace": self.execution_trace,
            }

        self.add_trace(
            step="Model Selection",
            status="completed",
            details=(
                f"Selected model: "
                f"{model_info['name']}"
            ),
        )

        # ====================================================
        # STEP 9: MODEL LOADING
        # ====================================================

        try:
            model = get_model(task)

        except Exception as error:

            self.add_trace(
                step="Model Loading",
                status="failed",
                details=str(error),
            )

            return {
                "success": False,
                "task": task,
                "task_description": task_description,
                "model": model_info["name"],
                "routing": routing_decision,
                "error": str(error),
                "trace": self.execution_trace,
            }

        self.add_trace(
            step="Model Loading",
            status="completed",
            details=(
                f"{model_info['name']} "
                "loaded successfully."
            ),
        )

        # ====================================================
        # STEP 10: MODEL EXECUTION
        # ====================================================

        try:

            # ------------------------------------------------
            # IMPORTANT:
            # Keep the raw model output separately.
            # Do NOT overwrite it with the final answer.
            # ------------------------------------------------

            model_result = None
            answer = None

            # ------------------------------------------------
            # VQA
            # ------------------------------------------------

            if task == "vqa":

                model_result = model.predict(
                    image=images[0],
                    question=query,
                )

                answer = model_result

            # ------------------------------------------------
            # CAPTIONING
            # ------------------------------------------------

            elif task == "captioning":

                model_result = model.caption(
                    image=images[0],
                )

                answer = model_result

            # ------------------------------------------------
            # CHANGE DETECTION
            # ------------------------------------------------

            elif task == "change_detection":

                model_result = model.predict(
                    image1=images[0],
                    image2=images[1],
                )

                if not model_result.get(
                    "success",
                    False,
                ):
                    raise RuntimeError(
                        model_result.get(
                            "error",
                            "Change detection failed.",
                        )
                    )

                change_percentage = model_result.get(
                    "change_percentage",
                    None,
                )

                image_size = model_result.get(
                    "image_size",
                    {},
                )

                width = image_size.get(
                    "width",
                    "unknown",
                )

                height = image_size.get(
                    "height",
                    "unknown",
                )

                answer = (
                    "Bi-temporal change detection completed. "
                    f"Approximately {change_percentage}% "
                    "of the analyzed pixels were identified "
                    f"as changed. Analysis size: "
                    f"{width} × {height} pixels."
                )

            # ------------------------------------------------
            # CHANGE-VQA
            # ------------------------------------------------

            elif task == "change_vqa":

                model_result = model.predict(
                    image_before=images[0],
                    image_after=images[1],
                    question=query,
                )

                answer = model_result

            # ------------------------------------------------
            # OPTICAL + SAR
            # ------------------------------------------------

            elif task == "optical_sar":

                model_result = model.predict(
                    optical_image=images[0],
                    sar_image=images[1],
                )

                if not model_result.get(
                    "success",
                    False,
                ):
                    raise RuntimeError(
                        model_result.get(
                            "error",
                            "Optical-SAR analysis failed.",
                        )
                    )

                aligned_size = model_result.get(
                    "aligned_size",
                    {},
                )

                width = aligned_size.get(
                    "width",
                    "unknown",
                )

                height = aligned_size.get(
                    "height",
                    "unknown",
                )

                answer = (
                    "Optical and SAR imagery were "
                    "successfully processed and fused. "
                    f"Aligned analysis size: "
                    f"{width} × {height} pixels."
                )

            else:

                raise ValueError(
                    f"Unsupported task: {task}"
                )

        except Exception as error:

            self.add_trace(
                step="Model Execution",
                status="failed",
                details=str(error),
            )

            return {
                "success": False,
                "task": task,
                "task_description": task_description,
                "model": model_info["name"],
                "routing": routing_decision,
                "error": str(error),
                "trace": self.execution_trace,
            }

        # ====================================================
        # STEP 11: MODEL EXECUTION TRACE
        # ====================================================

        self.add_trace(
            step="Model Execution",
            status="completed",
            details=(
                "Specialist model executed "
                "successfully."
            ),
        )

        # ====================================================
        # STEP 12: STANDARDIZED EVIDENCE
        # ====================================================

        evidence = create_multi_image_evidence(
            images
        )

        confidence = None
        confidence_type = "unavailable"

        # ----------------------------------------------------
        # CHANGE DETECTION
        # ----------------------------------------------------

        if task == "change_detection":

            change_map_path = model_result.get(
                "change_map"
            )

            if change_map_path:

                evidence.append(
                    create_change_map_evidence(
                        change_map_path
                    )
                )

            confidence = model_result.get(
                "confidence"
            )

            confidence_type = model_result.get(
                "confidence_type",
                "heuristic_diagnostic",
            )

        # ----------------------------------------------------
        # OPTICAL + SAR
        # ----------------------------------------------------

        elif task == "optical_sar":

            fusion_path = model_result.get(
                "evidence"
            )

            if fusion_path:

                evidence.append(
                    create_optical_sar_evidence(
                        fusion_path
                    )
                )

            confidence = model_result.get(
                "confidence"
            )

            confidence_type = model_result.get(
                "confidence_type",
                "heuristic_diagnostic",
            )

        # ----------------------------------------------------
        # VQA / CAPTIONING / CHANGE-VQA
        # ----------------------------------------------------

        elif task in [
            "vqa",
            "captioning",
            "change_vqa",
        ]:

            if isinstance(
                model_result,
                dict,
            ):

                confidence = model_result.get(
                    "confidence"
                )

                confidence_type = model_result.get(
                    "confidence_type",
                    "unavailable",
                )

        self.add_trace(
            step="Evidence Generation",
            status="completed",
            details=(
                f"{len(evidence)} evidence item(s) "
                "generated."
            ),
        )

        # ====================================================
        # STEP 13: RESULT GENERATION
        # ====================================================

        self.add_trace(
            step="Result Generation",
            status="completed",
            details="Answer generated successfully.",
        )

        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        return {

            "success": True,

            "task": task,

            "task_description":
                task_description,

            "model":
                model_info["name"],

            "model_status":
                "loaded",

            # ------------------------------------------------
            # AGENT INFORMATION
            # ------------------------------------------------

            "routing":
                routing_decision,

            "routing_confidence":
                routing_confidence,

            "routing_reason":
                routing_reason,

            "candidate_scores":
                candidate_scores,

            # ------------------------------------------------
            # INPUT INFORMATION
            # ------------------------------------------------

            "modalities":
                modality_results,

            "compatibility":
                compatibility,

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            "answer":
                answer,

            "evidence":
                evidence,

            "confidence":
                confidence,

            "confidence_type":
                confidence_type,

            # ------------------------------------------------
            # EXECUTION TRACE
            # ------------------------------------------------

            "trace":
                self.execution_trace,
        }


# ============================================================
# CONTROLLER INSTANCE
# ============================================================

controller = SatQueryController()