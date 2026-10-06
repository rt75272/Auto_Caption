"""
OCR and price-label helpers for wide shelf-bay images.

This module converts Tesseract results into reusable word records and uses
those records for two conservative checks: locating a unique shelf price as
a candidate anchor, and rejecting a vision-model box that contains readable
product packaging. The unique-price estimate is never treated as proof of an
empty slot; it still passes the independent visual vacancy check in
placement.py.

Provides:
    read_bay_text: OCR one bay and cache each word's text, confidence, and
        pixel rectangle.
    find_empty_slot_box: estimate a candidate product area above a unique
        matching shelf-price label.
    box_contains_product_text: detect readable merchandise text inside a
        proposed box.
    find_price_label_word: identify the unique OCR word matching a tag price.
    BayTextReader: hold an OCR cache for one scan.

Used by: placement.py and pipeline.py.
"""

import re
from typing import Dict, List, MutableMapping, Tuple

import pytesseract
from PIL import Image

from .config import TITLE_METADATA_WORDS
from .models import OcrWord


def read_bay_text(image_path, cache):
    """
    OCR visible text and cache each word's confidence and image coordinates.

    Args:
        image_path: Path to a bay image.
        cache: Mutable mapping used to reuse OCR results by image path.

    Returns:
        A list of normalized text, confidence, bounding-box coordinates,
        and original text tuples for the image.
    """
    if image_path not in cache:
        image = Image.open(image_path).convert("RGB")
        data = pytesseract.image_to_data(
            image,
            config="--oem 3 --psm 11",
            output_type=pytesseract.Output.DICT,
        )
        cache[image_path] = [
            OcrWord(
                re.sub(r"[^A-Z0-9]", "", text.upper()),
                float(data["conf"][index]),
                int(data["left"][index]),
                int(data["top"][index]),
                int(data["width"][index]),
                int(data["height"][index]),
                text.strip(),
            )
            for index, text in enumerate(data["text"])
            if text.strip()
        ]
    return cache[image_path]

def find_empty_slot_box(price, bay_words, image_size):
    """
    Estimate a vacant shelf slot from its unique matching price label.

    Title OCR is deliberately not used here: matching printed text on a
    product could incorrectly mark an in-stock package as out of stock.

    Args:
        price: Current price read from the close-up tag, without a currency
            symbol.
        bay_words: OCR word records returned by read_bay_text.
        image_size: Bay image dimensions as (width, height), in pixels.

    Returns:
        A pixel rectangle (left, top, right, bottom) immediately above the
        unique matching shelf-price label, or None if the price is missing or
        ambiguous.
    """
    if price:
        normalized_price = re.sub(r"\D", "", price)
        price_hits = [
            word
            for word in bay_words
            if re.sub(r"\D", "", word[6]) == normalized_price
            and word[1] >= 35
        ]
        if len(price_hits) == 1:
            _, confidence, left, top, width, _, _ = price_hits[0]
            if confidence < 35:
                return None

            center_x = left + width / 2
            box_width = image_size[0] * 0.12
            box_height = image_size[1] * 0.14
            bottom = max(0, top - image_size[1] * 0.015)
            return (
                max(0, center_x - box_width / 2),
                max(0, bottom - box_height),
                min(image_size[0], center_x + box_width / 2),
                bottom,
            )

    return None

def box_contains_product_text(product_box, bay_words):
    """
    Detect readable merchandise text substantially inside a candidate box.

    Args:
        product_box: Pixel rectangle in (left, top, right, bottom) order.
        bay_words: OCR word records returned by read_bay_text.

    Returns:
        True when multiple readable alphabetic words or one strong product
        word overlap the box, indicating that merchandise may occupy it.
    """
    left, top, right, bottom = product_box
    readable_words = []
    for word in bay_words:
        _, confidence, word_left, word_top, word_width, word_height, raw = word
        letters = re.sub(r"[^A-Z]", "", raw.upper())
        if confidence < 45 or len(letters) < 2:
            continue
        if letters in TITLE_METADATA_WORDS or not word_width or not word_height:
            continue

        overlap_width = max(
            0,
            min(right, word_left + word_width) - max(left, word_left),
        )
        overlap_height = max(
            0,
            min(bottom, word_top + word_height) - max(top, word_top),
        )
        word_area = word_width * word_height
        if overlap_width * overlap_height / word_area >= 0.35:
            readable_words.append((letters, confidence))

    unique_words = {letters for letters, _ in readable_words}
    return len(unique_words) >= 2 or any(
        len(letters) >= 5 and confidence >= 70
        for letters, confidence in readable_words
    )

def find_price_label_word(price, bay_words):
    """
    Find a unique OCR word matching a shelf price, allowing one digit error.

    Args:
        price: Current price read from the close-up tag.
        bay_words: OCR word records returned by read_bay_text.

    Returns:
        The matching OCR word record, or None if the price is missing,
        ambiguous, or does not have a unique high-confidence near-match.
    """
    normalized_price = re.sub(r"\D", "", price or "")
    if not normalized_price:
        return None

    exact_hits = [
        word
        for word in bay_words
        if re.sub(r"\D", "", word[6]) == normalized_price
        and word[1] >= 35
    ]
    if len(exact_hits) == 1:
        return exact_hits[0]
    if exact_hits:
        return None

    near_hits = [
        word
        for word in bay_words
        if word[1] >= 80
        and len(re.sub(r"\D", "", word[6])) == len(normalized_price)
        and sum(
            left != right
            for left, right in zip(
                re.sub(r"\D", "", word[6]),
                normalized_price,
            )
        )
        == 1
    ]
    return near_hits[0] if len(near_hits) == 1 else None

class BayTextReader:
    """Read and cache OCR words for bay images in a single scan.

    Attributes:
        cache: Mapping from image paths to their recognized words.
    """

    def __init__(self) -> None:
        """Create an empty per-run OCR cache.

        Args:
            None.

        Returns:
            None.
        """
        self.cache = {}

    def read(self, image_path: str):
        """Get OCR words for a bay, computing and caching them if needed.

        Args:
            image_path: Path to a wide bay image.

        Returns:
            Recognized words with confidence and pixel geometry.
        """
        return read_bay_text(image_path, self.cache)
