from agent.controller import SatQueryController


def main():
    print("=" * 60)
    print("SATQUERY AI - OPTICAL + SAR CONTROLLER TEST")
    print("=" * 60)

    controller = SatQueryController()

    result = controller.analyze(
    query="Analyze these optical and SAR images together.",
    images=[
        "data/samples/test.jpg",
        "data/samples/test.tiff",
    ],
    modalities=[
        "optical",
        "sar",
    ],
)

    print("\nSuccess:")
    print(result.get("success"))

    print("\nTask:")
    print(result.get("task"))

    print("\nTask Description:")
    print(result.get("task_description"))

    print("\nModel:")
    print(result.get("model"))

    print("\nModel Status:")
    print(result.get("model_status"))

    print("\nAnswer:")
    print(result.get("answer"))

    print("\nEvidence:")
    print(result.get("evidence"))

    print("\nConfidence:")
    print(result.get("confidence"))

    print("\nExecution Trace:")
    print("-" * 60)

    for step in result.get("trace", []):
        print(
            f"[{step.get('status', '').upper()}] "
            f"{step.get('step', '')}: "
            f"{step.get('message', '')}"
        )


if __name__ == "__main__":
    main()