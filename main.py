"""
Command-line entry point for the shelf-scanning application.

Run this file from the repository root with ``python main.py``. By default,
it recursively scans ``./pictures``, skips the example reference image, and
writes confirmed yellow-outlined bay copies to ``pictures/annotated_bays``.
The input folder can be overridden with one positional directory argument.
"""

import argparse
import sys

from shelf_scanner import __version__
from shelf_scanner.config import DEFAULT_INPUT_DIRECTORY
from shelf_scanner.pipeline import ShelfScanner
from shelf_scanner import console


def main() -> int:
    """Parse the optional input folder and run the shelf scanner.

    Args:
        None. Command-line arguments are read from the current process.

    Returns:
        Process exit status: zero after a completed scan and 130 after Ctrl-C.
    """
    parser = argparse.ArgumentParser(
        description="Find and outline confirmed empty shelf slots."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=DEFAULT_INPUT_DIRECTORY,
        help=(
            "root folder containing bay and MACRO_FOCUS images "
            f"(default: {DEFAULT_INPUT_DIRECTORY})"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"shelf-scanner {__version__}",
    )
    arguments = parser.parse_args()
    try:
        ShelfScanner().run(arguments.folder)
    except KeyboardInterrupt:
        console.warning(
            "Processing interrupted. Any fully written bay annotations "
            "remain in annotated_bays."
        )
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
