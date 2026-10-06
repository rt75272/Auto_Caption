"""
Rectangle geometry and vision-model coordinate helpers.

Several stages of the program describe regions of a photo as rectangles: the
bounding box of an OCR word, a candidate empty shelf slot proposed by the
vision model, and the box finally drawn on the bay photo. This module gives
those rectangles one shared representation and the few operations the other
modules need.

The vision model reports coordinates on a 0..1000 grid that is independent of
the real image size, so parse_model_box converts them to fractions of the
image (0..1) while rejecting values that cannot be a position in the photo.
Callers then scale the fractions by the image width and height to get pixels.

Provides:
    Box: a (left, top, right, bottom) rectangle with width, height, center,
        overlap, union, rounding, and clamping helpers.
    parse_model_box: convert model coordinates to a validated 0..1 box.

Used by: bay_text, placement, models, renderer, and pipeline.
"""

from typing import NamedTuple, Optional, Sequence, Tuple


class Box(NamedTuple):
    """
    Axis-aligned rectangle stored as (left, top, right, bottom).

    Because a Box is a tuple it unpacks, indexes, and compares like the plain
    rectangles Pillow expects, so it can be passed straight to drawing calls.

    Attributes:
        left: X coordinate of the left edge.
        top: Y coordinate of the top edge.
        right: X coordinate of the right edge.
        bottom: Y coordinate of the bottom edge.
    """

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        """
        Horizontal extent of the box.

        Args:
            None.

        Returns:
            The distance from the left edge to the right edge.
        """
        return self.right - self.left

    @property
    def height(self) -> float:
        """
        Vertical extent of the box.

        Args:
            None.

        Returns:
            The distance from the top edge to the bottom edge.
        """
        return self.bottom - self.top

    @property
    def center(self) -> Tuple[float, float]:
        """
        Middle point of the box.

        Args:
            None.

        Returns:
            The (x, y) coordinates of the center.
        """
        return (self.left + self.right) / 2, (self.top + self.bottom) / 2

    def overlap_area(self, other: "Box") -> float:
        """
        Measure how much two boxes overlap.

        Args:
            other: The box to intersect with this one.

        Returns:
            The intersection area, or 0 when the boxes do not overlap.
        """
        overlap_width = max(
            0, min(self.right, other.right) - max(self.left, other.left)
        )
        overlap_height = max(
            0, min(self.bottom, other.bottom) - max(self.top, other.top)
        )
        return overlap_width * overlap_height

    def union(self, other: "Box") -> "Box":
        """
        Find the smallest box that contains both boxes.

        Args:
            other: The box to merge with this one.

        Returns:
            A box spanning the extremes of both inputs.
        """
        return Box(
            min(self.left, other.left),
            min(self.top, other.top),
            max(self.right, other.right),
            max(self.bottom, other.bottom),
        )

    def rounded(self) -> "Box":
        """
        Round every edge to the nearest whole pixel.

        Returns:
            A box whose coordinates are integers.
        """
        return Box(*(int(round(value)) for value in self))

    def clamped(self, size: Tuple[int, int]) -> "Box":
        """
        Restrict the box to an image.

        Args:
            size: Image dimensions as (width, height), in pixels.

        Returns:
            A box whose edges lie within the image bounds.
        """
        width, height = size
        return Box(
            max(0, min(self.left, width)),
            max(0, min(self.top, height)),
            max(0, min(self.right, width)),
            max(0, min(self.bottom, height)),
        )


def parse_model_box(
    values: Sequence[float], tolerance: float = 0.0
) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert vision-model box coordinates to fractions of the image.

    The model reports 0..1000 coordinates, but 0..1 values are accepted too.

    Args:
        values: Four numbers ordered left, top, right, bottom.
        tolerance: How far outside 0..1 a converted value may fall before the
            box is rejected. Accepted values are then clamped into 0..1.

    Returns:
        A (left, top, right, bottom) tuple within 0..1, or None when the
        input is not four numbers or lies beyond the tolerance.
    """
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    try:
        coordinates = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if max(coordinates) > 1:
        coordinates = [value / 1000 for value in coordinates]
    if any(
        value < -tolerance or value > 1 + tolerance for value in coordinates
    ):
        return None
    left, top, right, bottom = (
        max(0.0, min(1.0, value)) for value in coordinates
    )
    return left, top, right, bottom
