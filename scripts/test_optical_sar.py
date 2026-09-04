from models.optical_sar import get_optical_sar_model


OPTICAL_IMAGE = "data/samples/test.jpg"
SAR_IMAGE = "data/samples/test.tiff"


def main():

    print("=" * 60)
    print("SATQUERY AI - OPTICAL + SAR TEST")
    print("=" * 60)

    model = get_optical_sar_model()

    result = model.predict(
        optical_image=OPTICAL_IMAGE,
        sar_image=SAR_IMAGE
    )

    print("\nOptical + SAR Result")
    print("=" * 60)

    for key, value in result.items():

        # Don't print the entire fused array
        if key == "fused_output":
            print(
                f"{key}: "
                f"array shape = {value.shape}"
            )
        else:
            print(
                f"{key}: {value}"
            )


if __name__ == "__main__":
    main()