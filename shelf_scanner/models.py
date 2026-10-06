"""
Plain data containers passed between the stages of the pipeline.

The program is a chain of stages, and each hands a small, typed record to the
next instead of loose tuples and dictionaries:

    OcrWord     one word Tesseract found in a bay photo, with its confidence
                and pixel position. Produced by bay_text, consumed by the
                locating logic to find price labels and to detect text that
                proves a region holds a stocked product.

    TagInfo     the product title and price read from one close-up tag.
                Produced by tag_reader.

    Annotation  the final answer for one tag: which bay it belongs to, its
                title, and the pixel box of the empty slot (or None when no
                empty slot could be confirmed). Produced by pipeline and
                drawn by renderer.

OcrWord is a NamedTuple so it still unpacks like the plain tuples Tesseract
results were originally stored as; the other two are ordinary dataclasses.

Used by: bay_text, tag_reader, placement, pipeline, and renderer.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

from .geometry import Box


class OcrWord(NamedTuple):
    """
    One word recognized by Tesseract in a bay photo.

    Attributes:
        text: The word reduced to uppercase letters and digits.
        confidence: Tesseract confidence from 0 to 100.
        left: X coordinate of the word's left edge, in pixels.
        top: Y coordinate of the word's top edge, in pixels.
        width: Word width in pixels.
        height: Word height in pixels.
        raw: The word exactly as recognized, including punctuation.
    """

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    raw: str

    @property
    def box(self) -> Box:
        """
        Pixel rectangle occupied by the word.

        Args:
            None.

        Returns:
            A Box covering the word's recognized extent.
        """
        return Box(
            self.left,
            self.top,
            self.left + self.width,
            self.top + self.height,
        )


@dataclass
class TagInfo:
    """
    Text read from a close-up shelf tag.

    Attributes:
        title: Product title, or None when nothing could be read.
        price: Current price without a currency symbol, or None.
    """

    title: Optional[str]
    price: Optional[str]


@dataclass
class Annotation:
    """
    The result of pairing one close-up tag with its bay photo.

    Attributes:
        bay_path: Path to the wide bay image that will be annotated.
        tag_path: Path to the close-up tag image.
        title: Product title read from the tag.
        product_box: Pixel rectangle of the empty slot, or None when no
            visibly empty slot could be confirmed.
    """

    bay_path: str
    tag_path: str
    title: str
    product_box: Optional[Box]

    @property
    def is_located(self) -> bool:
        """
        Report whether an empty slot was confirmed.

        Args:
            None.

        Returns:
            True when a product box is available to draw.
        """
        return self.product_box is not None
