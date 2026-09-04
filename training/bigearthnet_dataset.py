import os
import subprocess
from collections import defaultdict


ARCHIVE = r"D:\RISHIT 2\Downloads\BigEarthNet-S2.tar.zst"


def check_archive():

    if not os.path.exists(ARCHIVE):
        print("Archive not found:")
        print(ARCHIVE)
        return False

    size_gb = os.path.getsize(ARCHIVE) / (1024 ** 3)

    print("=" * 70)
    print("BIGEARTHNET-S2 INSPECTION")
    print("=" * 70)

    print(f"Archive: {ARCHIVE}")
    print(f"Size: {size_gb:.2f} GB")

    return True


def list_entries():

    print("\nReading archive contents...")

    command = [
        "tar",
        "-tf",
        ARCHIVE
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    patches = defaultdict(list)

    count = 0

    for line in process.stdout:

        line = line.strip()

        if not line:
            continue

        count += 1

        parts = line.split("/")

        if len(parts) < 4:
            continue

        patch_name = parts[-2]
        filename = parts[-1]

        if filename.lower().endswith(".tif"):

            patches[patch_name].append(filename)

        if count >= 5000:
            break

    process.kill()

    print("\nSample patches found:")
    print("-" * 70)

    shown = 0

    for patch, files in patches.items():

        print(f"\nPatch: {patch}")

        for file in sorted(files):
            print(f"  {file}")

        shown += 1

        if shown >= 10:
            break

    print("\nTotal archive entries inspected:", count)


def main():

    if not check_archive():
        return

    list_entries()


if __name__ == "__main__":
    main()