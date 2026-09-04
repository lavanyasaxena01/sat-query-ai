import os
import numpy as np
import rasterio
from PIL import Image


class BiTemporalChangeDetector:
    """
    Baseline bi-temporal change detection model.

    Input:
        image1 = earlier satellite image
        image2 = later satellite image

    Output:
        binary change map
        change percentage
        confidence
    """

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    # ========================================================
    # IMAGE LOADING
    # ========================================================

    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load a satellite image.

        Supports:
            GeoTIFF
            TIFF
            PNG
            JPEG
        """

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        extension = os.path.splitext(
            image_path
        )[1].lower()

        # ----------------------------------------------------
        # GeoTIFF / TIFF
        # ----------------------------------------------------

        if extension in [".tif", ".tiff"]:

            with rasterio.open(image_path) as src:

                image = src.read()

                # Convert:
                # bands, height, width
                #
                # to:
                # height, width, bands

                image = np.transpose(
                    image,
                    (1, 2, 0)
                )

        # ----------------------------------------------------
        # PNG / JPEG
        # ----------------------------------------------------

        elif extension in [
            ".png",
            ".jpg",
            ".jpeg",
        ]:

            image = np.array(
                Image.open(image_path)
            )

            if image.ndim == 2:
                image = image[..., np.newaxis]

        else:

            raise ValueError(
                f"Unsupported image format: {extension}"
            )

        return image.astype(np.float32)

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize image values to [0, 1].
        """

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        minimum = np.min(image)
        maximum = np.max(image)

        if maximum == minimum:
            return np.zeros_like(image)

        image = (
            image - minimum
        ) / (
            maximum - minimum
        )

        return image

    # ========================================================
    # SHAPE ALIGNMENT
    # ========================================================

    def _align_shapes(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
    ):
        """
        Make the two images spatially compatible
        at the array level.

        NOTE:
        Proper geospatial co-registration will be handled
        later in preprocessing/validation.
        """

        height = min(
            image1.shape[0],
            image2.shape[0],
        )

        width = min(
            image1.shape[1],
            image2.shape[1],
        )

        image1 = image1[
            :height,
            :width,
        ]

        image2 = image2[
            :height,
            :width,
        ]

        return image1, image2

    # ========================================================
    # BAND ALIGNMENT
    # ========================================================

    def _align_bands(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
    ):
        """
        Use the common number of bands.
        """

        bands = min(
            image1.shape[-1],
            image2.shape[-1],
        )

        image1 = image1[..., :bands]
        image2 = image2[..., :bands]

        return image1, image2

    # ========================================================
    # CHANGE CALCULATION
    # ========================================================

    def _calculate_change(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate normalized pixel-level difference.
        """

        difference = np.abs(
            image2 - image1
        )

        # Mean difference across bands

        change_score = np.mean(
            difference,
            axis=-1,
        )

        return change_score

    # ========================================================
    # CHANGE MAP
    # ========================================================

    def _create_change_map(
        self,
        change_score: np.ndarray,
    ) -> np.ndarray:
        """
        Convert change scores into a binary map.
        """

        change_map = (
            change_score >= self.threshold
        ).astype(np.uint8)

        return change_map

    # ========================================================
    # SAVE CHANGE MAP
    # ========================================================

    def _save_change_map(
        self,
        change_map: np.ndarray,
        output_path: str,
    ):
        """
        Save binary change map as PNG.
        """

        output_image = (
            change_map * 255
        ).astype(np.uint8)

        Image.fromarray(
            output_image
        ).save(output_path)

    # ========================================================
    # MAIN PREDICTION
    # ========================================================

    def predict(
        self,
        image1: str,
        image2: str,
        output_path: str = "outputs/change_map.png",
    ) -> dict:
        """
        Perform bi-temporal change detection.
        """

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        earlier = self._load_image(
            image1
        )

        later = self._load_image(
            image2
        )

        # ----------------------------------------------------
        # ALIGN
        # ----------------------------------------------------

        earlier, later = self._align_shapes(
            earlier,
            later,
        )

        earlier, later = self._align_bands(
            earlier,
            later,
        )

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        earlier = self._normalize(
            earlier
        )

        later = self._normalize(
            later
        )

        # ----------------------------------------------------
        # CHANGE SCORE
        # ----------------------------------------------------

        change_score = self._calculate_change(
            earlier,
            later,
        )

        # ----------------------------------------------------
        # CHANGE MAP
        # ----------------------------------------------------

        change_map = self._create_change_map(
            change_score
        )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        total_pixels = change_map.size

        changed_pixels = int(
            np.sum(change_map)
        )

        change_percentage = (
            changed_pixels
            / total_pixels
            * 100
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = float(
            np.mean(
                np.abs(
                    change_score
                    - self.threshold
                )
            )
        )

        confidence = min(
            confidence * 2,
            1.0,
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output_directory = os.path.dirname(
            output_path
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        self._save_change_map(
            change_map,
            output_path,
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return {
            "success": True,

            "task":
                "change_detection",

            "change_map":
                output_path,

            "image_size": {
                "width":
                    int(change_map.shape[1]),

                "height":
                    int(change_map.shape[0]),
            },

            "changed_pixels":
                changed_pixels,

            "total_pixels":
                int(total_pixels),

            "change_percentage":
                round(
                    float(change_percentage),
                    2,
                ),

            "confidence":
    round(
        float(confidence),
        4,
    ),

"confidence_type":
    "heuristic_diagnostic",

}


# ============================================================
# SINGLETON
# ============================================================

_model_instance = None


def get_change_detection_model():
    """
    Return a shared change-detection instance.
    """

    global _model_instance

    if _model_instance is None:

        _model_instance = (
            BiTemporalChangeDetector()
        )

    return _model_instance