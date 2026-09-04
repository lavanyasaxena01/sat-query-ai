import os
import shutil
import subprocess
import tempfile

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
import torch
from torch.utils.data import Dataset


ARCHIVE = r"D:\RISHIT 2\Downloads\BigEarthNet-S2.tar.zst"


class BigEarthNetDirect(Dataset):

    BANDS = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]

    def __init__(
        self,
        archive_path,
        max_patches=1,
        image_size=120,
    ):

        self.archive_path = archive_path
        self.max_patches = max_patches
        self.image_size = image_size

        if not os.path.exists(self.archive_path):
            raise FileNotFoundError(
                f"Archive not found: {self.archive_path}"
            )

        print("Building patch index...")

        self.patches = self._build_patch_index()

        print(
            f"Found {len(self.patches)} patches."
        )

    # ---------------------------------------------------------
    # BUILD PATCH INDEX
    # ---------------------------------------------------------

    def _build_patch_index(self):

        command = [
            "tar",
            "-tf",
            self.archive_path
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        patches = {}

        for line in process.stdout:

            line = line.strip()

            if not line.endswith(".tif"):
                continue

            parts = line.split("/")

            if len(parts) < 3:
                continue

            patch_name = parts[-2]

            if patch_name not in patches:

                patches[patch_name] = {
                    "name": patch_name,
                    "files": []
                }

            patches[patch_name]["files"].append(line)

        process.wait()

        complete = []

        for patch in patches.values():

            if len(patch["files"]) >= 12:

                patch["files"] = sorted(
                    patch["files"]
                )

                complete.append(patch)

            if len(complete) >= self.max_patches:
                break

        return complete

    # ---------------------------------------------------------
    # LENGTH
    # ---------------------------------------------------------

    def __len__(self):

        return len(self.patches)

    # ---------------------------------------------------------
    # EXTRACT COMPLETE PATCH
    # ---------------------------------------------------------

    def _extract_patch(self, patch):

        temp_dir = tempfile.mkdtemp(
            prefix="bigearthnet_"
        )

        command = [
            "tar",
            "-xf",
            self.archive_path,
            "-C",
            temp_dir,
        ]

        command.extend(
            patch["files"]
        )

        print(
            f"\nExtracting patch: {patch['name']}"
        )

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )

            raise RuntimeError(
                result.stderr
            )

        return temp_dir

    # ---------------------------------------------------------
    # FIND EXTRACTED FILE
    # ---------------------------------------------------------

    def _find_extracted_file(
        self,
        temp_dir,
        archive_path
    ):

        # Archive path:
        #
        # BigEarthNet-S2/
        #     patch/
        #         image.tif

        relative = archive_path

        direct_path = os.path.join(
            temp_dir,
            relative
        )

        if os.path.exists(direct_path):
            return direct_path

        # Fallback search
        filename = os.path.basename(
            archive_path
        )

        for root, _, files in os.walk(
            temp_dir
        ):

            if filename in files:

                return os.path.join(
                    root,
                    filename
                )

        raise FileNotFoundError(
            f"Could not locate extracted file: "
            f"{archive_path}"
        )

    # ---------------------------------------------------------
    # LOAD + RESAMPLE BAND
    # ---------------------------------------------------------

    def _load_band(
        self,
        path
    ):

        with rasterio.open(path) as src:

            data = src.read(
                1
            ).astype(
                np.float32
            )

            # Resample directly to common grid
            resized = src.read(
                1,
                out_shape=(
                    self.image_size,
                    self.image_size
                ),
                resampling=Resampling.bilinear
            ).astype(
                np.float32
            )

        return resized

    # ---------------------------------------------------------
    # NORMALIZE BAND
    # ---------------------------------------------------------

    def _normalize(
        self,
        image
    ):

        image = image.astype(
            np.float32
        )

        valid = np.isfinite(
            image
        )

        if not np.any(valid):

            return np.zeros_like(
                image,
                dtype=np.float32
            )

        values = image[valid]

        low = np.percentile(
            values,
            2
        )

        high = np.percentile(
            values,
            98
        )

        if high <= low:

            return np.zeros_like(
                image,
                dtype=np.float32
            )

        image = np.clip(
            image,
            low,
            high
        )

        image = (
            image - low
        ) / (
            high - low
        )

        return image.astype(
            np.float32
        )

    # ---------------------------------------------------------
    # LOAD COMPLETE PATCH
    # ---------------------------------------------------------

    def _load_patch(
        self,
        patch
    ):

        temp_dir = self._extract_patch(
            patch
        )

        try:

            arrays = []

            for archive_path in patch["files"]:

                local_path = (
                    self._find_extracted_file(
                        temp_dir,
                        archive_path
                    )
                )

                band = self._load_band(
                    local_path
                )

                band = self._normalize(
                    band
                )

                arrays.append(
                    band
                )

            image = np.stack(
                arrays,
                axis=0
            )

            return image

        finally:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )

    # ---------------------------------------------------------
    # GET ITEM
    # ---------------------------------------------------------

    def __getitem__(
        self,
        index
    ):

        patch = self.patches[index]

        image = self._load_patch(
            patch
        )

        image = torch.from_numpy(
            image
        ).float()

        return {
            "image": image,
            "patch_name": patch["name"]
        }


# =============================================================
# TEST
# =============================================================

def main():

    dataset = BigEarthNetDirect(
        archive_path=ARCHIVE,
        max_patches=1,
        image_size=120
    )

    print("\nDataset test")
    print("=" * 60)

    print(
        "Number of patches:",
        len(dataset)
    )

    sample = dataset[0]

    image = sample["image"]

    print(
        "\nPatch:",
        sample["patch_name"]
    )

    print(
        "Tensor shape:",
        image.shape
    )

    print(
        "Minimum:",
        image.min().item()
    )

    print(
        "Maximum:",
        image.max().item()
    )

    print(
        "Mean:",
        image.mean().item()
    )


if __name__ == "__main__":
    main()