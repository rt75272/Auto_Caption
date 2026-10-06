"""
Image discovery, scene classification, and tag-to-bay pairing.

The input may contain nested folders (for example, pictures/training). This
catalog walks them in a stable order, skips the supplied example image and
previous output folder, and classifies close-up tags by their MACRO_FOCUS
filename marker. Other images are considered bay shots only when ORB feature
points cover enough of the image. A tag is paired with the most recent bay
captured before it, matching the capture sequence used for this dataset.

Provides:
    ImageGroups: discovered files grouped into tag and bay paths.
    ImageCatalog.discover: recursively find images and classify their roles.
    capture_timestamp, has_wide_scene_coverage, find_matching_bay: focused
        helpers used by discovery and pairing.

Used by: pipeline.py.
"""

import os
import re
from datetime import datetime
from typing import List, NamedTuple, Optional

import cv2
from tqdm import tqdm

from . import console
from .config import (
    IMAGE_EXTENSIONS,
    MAX_BAY_AGE_SECONDS,
    MIN_BAY_GRID_COVERAGE,
    OUTPUT_DIRECTORY_NAME,
    REFERENCE_IMAGE_STEM,
    SCENE_GRID_SIZE,
    TAG_FILENAME_MARKER,
)


class ImageGroups(NamedTuple):
    """Image paths found by the catalog.

    Attributes:
        all_files: Every supported input image except the reference image and
            images in the generated output folder.
        tag_paths: Close-up shelf-tag photos marked MACRO_FOCUS.
        bay_paths: Wide shelf scenes suitable for pairing with those tags.
    """

    all_files: List[str]
    tag_paths: List[str]
    bay_paths: List[str]


def capture_timestamp(path):
    """
    Parse the camera timestamp embedded in an image filename.

    Args:
        path: Image path containing a YYYYMMDD_HHMMSSmmm timestamp.

    Returns:
        A datetime for the timestamp, or None if the filename has no match.
    """
    match = re.search(r"(\d{8})_(\d{9})", os.path.basename(path))
    if match is None:
        return None
    return datetime.strptime(
        match.group(1) + match.group(2)[:6],
        "%Y%m%d%H%M%S",
    )

def has_wide_scene_coverage(keypoints, image_shape):
    """
    Decide whether keypoints cover enough grid cells to indicate a bay scene.

    Args:
        keypoints: ORB keypoints detected in the image.
        image_shape: Grayscale image shape as (height, width).

    Returns:
        True when occupied grid cells meet MIN_BAY_GRID_COVERAGE.
    """
    occupied_cells = {
        (
            min(
                SCENE_GRID_SIZE - 1,
                int(kp.pt[0] / image_shape[1] * SCENE_GRID_SIZE),
            ),
            min(
                SCENE_GRID_SIZE - 1,
                int(kp.pt[1] / image_shape[0] * SCENE_GRID_SIZE),
            ),
        )
        for kp in keypoints
    }
    return len(occupied_cells) >= MIN_BAY_GRID_COVERAGE

def find_matching_bay(tag_path, bay_paths):
    """
    Pair a close-up tag with the bay photographed immediately before it.

    Each bay is shot first and its out-of-stock tags follow it, so a tag
    belongs to the latest bay captured at or before the tag. A bay taken
    seconds after the tag starts the next group and is not a match.

    Args:
        tag_path: Path to the close-up tag image.
        bay_paths: Candidate wide bay-image paths.

    Returns:
        The latest bay captured within MAX_BAY_AGE_SECONDS before the tag.
        When none qualifies, the nearest bay by capture time. If filenames
        lack timestamps, the first candidate. None when there are no bays.
    """
    if not bay_paths:
        return None

    tag_time = capture_timestamp(tag_path)
    timed_bays = [
        path for path in bay_paths if capture_timestamp(path) is not None
    ]
    if tag_time is None or not timed_bays:
        return bay_paths[0]

    preceding_bays = [
        path
        for path in timed_bays
        if capture_timestamp(path) <= tag_time
        and (tag_time - capture_timestamp(path)).total_seconds()
        <= MAX_BAY_AGE_SECONDS
    ]
    if preceding_bays:
        return max(preceding_bays, key=capture_timestamp)
    return min(
        timed_bays,
        key=lambda path: abs(
            (capture_timestamp(path) - tag_time).total_seconds()
        ),
    )

class ImageCatalog:
    """Find, classify, and pair input shelf images.

    Attributes:
        orb: OpenCV ORB feature detector reused for all images in a scan.
    """

    def __init__(self) -> None:
        """Create the ORB detector used to distinguish bay scenes.

        Args:
            None.

        Returns:
            None.
        """
        self.orb = cv2.ORB_create(nfeatures=2000)

    def discover(self, folder_path: str) -> ImageGroups:
        """Recursively discover supported images and classify them.

        Args:
            folder_path: Root folder containing bay images and close-up tags.

        Returns:
            ImageGroups with all supported inputs, tag paths, and bay paths.

        Raises:
            FileNotFoundError: If folder_path does not exist or is not a
                directory.
        """
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Input image folder does not exist: {folder_path}"
            )

        all_files = []
        for current_root, directories, filenames in os.walk(folder_path):
            directories[:] = sorted(
                directory
                for directory in directories
                if directory.lower() != OUTPUT_DIRECTORY_NAME.lower()
            )
            for filename in sorted(filenames):
                stem, extension = os.path.splitext(filename)
                if (
                    extension.lower() in IMAGE_EXTENSIONS
                    and stem.lower() != REFERENCE_IMAGE_STEM.lower()
                ):
                    all_files.append(os.path.join(current_root, filename))
        all_files.sort()

        image_data = {}
        for path in tqdm(
            all_files,
            desc="Loading images",
            unit="image",
            colour="cyan",
        ):
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                console.warning(f"Could not decode image {path}; skipping it.")
                continue
            keypoints = self.orb.detect(image, None)
            image_data[path] = {"shape": image.shape, "keypoints": keypoints}

        tag_paths = sorted(
            path
            for path in image_data
            if TAG_FILENAME_MARKER in os.path.basename(path).lower()
        )
        tag_path_set = set(tag_paths)
        bay_paths = sorted(
            path
            for path, data in image_data.items()
            if path not in tag_path_set
            and has_wide_scene_coverage(data["keypoints"], data["shape"])
        )
        return ImageGroups(all_files, tag_paths, bay_paths)

    def find_matching_bay(
        self,
        tag_path: str,
        bay_paths: List[str],
    ) -> Optional[str]:
        """Select the bay shot that most plausibly precedes a close-up tag.

        Args:
            tag_path: Path to a close-up shelf-tag photo.
            bay_paths: Candidate wide bay-image paths.

        Returns:
            The latest bay taken within the configured window before the tag,
            or the fallback selected by find_matching_bay.
        """
        return find_matching_bay(tag_path, bay_paths)
