"""
Shared settings and constants for the shelf scanner.

Every tunable value that more than one module needs lives here, so the rest
of the package contains no magic numbers or environment lookups. The values
fall into four groups:

    Bay pairing:
        SCENE_GRID_SIZE and MIN_BAY_GRID_COVERAGE decide whether a photo is a
        wide shot of a whole shelf bay or a close-up, and MAX_BAY_AGE_SECONDS
        limits how long before a close-up its bay photo may have been taken.

    File discovery:
        Which extensions count as images, which filename marker identifies a
        close-up tag photo, which reference image to skip, and the name of the
        folder that receives the annotated output.

    Tag text:
        UNREADABLE_TITLE is the placeholder used when no title can be read,
        and TITLE_METADATA_WORDS lists generic shelf-tag wording (SALE,
        GLUTEN, ...) that is never part of a product name.

    Vision model:
        The Ollama server address and model name. Both can be overridden
        with the OLLAMA_HOST and AUTOCAPTION_VISION_MODEL environment
        variables without editing code.

Used by: every other module in the package.
Imports from the package: nothing, so it can never cause a circular import.
"""

import os

# A photo is treated as a wide bay shot when ORB keypoints (distinctive image
# corners) fall in at least MIN_BAY_GRID_COVERAGE of the SCENE_GRID_SIZE x
# SCENE_GRID_SIZE cells. Close-ups concentrate their detail in a few cells.
SCENE_GRID_SIZE = 8
MIN_BAY_GRID_COVERAGE = 40

# A close-up is paired with the latest bay photographed at most this many
# seconds before it; the photographer shoots a bay first, then its tags.
MAX_BAY_AGE_SECONDS = 180

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Lowercase text that marks a filename as a close-up out-of-stock tag photo.
TAG_FILENAME_MARKER = "macro_focus"

# Filename stem of the sample picture that shows the desired output. It is
# documentation, not input, so the scanner skips it.
REFERENCE_IMAGE_STEM = "example"

# Subfolder, inside the picture folder, that receives the annotated bays.
OUTPUT_DIRECTORY_NAME = "annotated_bays"
DEFAULT_INPUT_DIRECTORY = "./pictures"

UNREADABLE_TITLE = "TITLE UNREADABLE"

# Wording printed on shelf tags that describes the offer, not the product.
TITLE_METADATA_WORDS = {
    "EA",
    "FREE",
    "GLUTEN",
    "GMO",
    "NONGMO",
    "NONGMOPROJECT",
    "OUNCE",
    "OUNCES",
    "PRICE",
    "PROJECT",
    "REG",
    "REGULAR",
    "SALE",
    "SAVE",
    "THRU",
    "THROUGH",
}

# Where the local Ollama server listens and which vision model to ask. Both
# can be overridden from the shell, for example:
#   AUTOCAPTION_VISION_MODEL=qwen3-vl:8b python main.py
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("AUTOCAPTION_VISION_MODEL", "qwen3.5:latest")
