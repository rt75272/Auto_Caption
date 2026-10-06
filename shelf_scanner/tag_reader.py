"""
Reading the product title and price from close-up shelf-tag photos.
"""

import re
from typing import Optional, Tuple

import numpy as np
import pytesseract
from PIL import Image

from . import console
from .config import TITLE_METADATA_WORDS, UNREADABLE_TITLE
from .models import TagInfo
from .ollama_client import OllamaClient


def extract_tag_details(tag_image_path):
    """
    Extract the product title and current price shown on a close-up tag.

    Args:
        tag_image_path: Path to the close-up shelf-tag image.

    Returns:
        A tuple of (title, price). Either value is None when OCR cannot
        confidently identify it.
    """
    try:
        img = Image.open(tag_image_path)
        # Page-segmentation mode 11 finds sparse text scattered across the
        # layout, which suits a tag that mixes a title, price, and barcode.
        data = pytesseract.image_to_data(
            img,
            config="--oem 3 --psm 11",
            output_type=pytesseract.Output.DICT,
        )
        # words holds title candidates as (top, left, text, confidence);
        # price_candidates holds price-shaped tokens as (confidence, top, price).
        words = []
        price_candidates = []
        for index, text in enumerate(data["text"]):
            text = text.strip()
            try:
                confidence = float(data["conf"][index])
            except ValueError:
                continue
            title_word = re.sub(r"[^A-Za-z]", "", text).upper()
            price_match = re.fullmatch(r"\$?(\d{1,3})[.,](\d{2})", text)
            # A price looks like 12.99 (optionally with a leading $), must be
            # read with some confidence, and sits in the middle band of the
            # tag. The top band holds the title; the bottom band holds fine
            # print such as the regular price and the per-ounce price.
            if (
                price_match
                and float(data["conf"][index]) >= 30
                and img.height * 0.2 <= int(data["top"][index]) <= img.height * 0.7
            ):
                price = f"{price_match.group(1)}.{price_match.group(2)}"
                price_candidates.append(
                    (confidence, int(data["top"][index]), price)
                )

            # Title words are printed in capitals in the upper part of the
            # tag and are not generic tag text such as SALE or GLUTEN.
            if (
                confidence >= 35
                and len(text) >= 3
                and text.isupper()
                and any(character.isalpha() for character in text)
                and int(data["top"][index]) < img.height * 0.68
                and title_word not in TITLE_METADATA_WORDS
            ):
                words.append(
                    (
                        int(data["top"][index]),
                        int(data["left"][index]),
                        text,
                        confidence,
                    )
                )

        if not words:
            return None, None

        # Words whose tops are within 5% of the image height share a text
        # line. The title is the line whose words have the most combined OCR
        # confidence, so best_word is any word on that line.
        best_word = max(
            words,
            key=lambda word: sum(
                other[3]
                for other in words
                if abs(other[0] - word[0]) <= img.height * 0.05
            ),
        )
        line_words = [
            word for word in words
            if abs(word[0] - best_word[0]) <= img.height * 0.05
        ]
        # Order the title words. When they lie on a straight line (a least-
        # squares fit leaves under 0.5% of the image height as error) they are
        # read left to right. Otherwise the title wraps onto several lines or
        # the photo is skewed, so they are read top to bottom, then left to
        # right.
        if len(line_words) >= 3:
            x_positions = np.array([word[1] for word in line_words])
            y_positions = np.array([word[0] for word in line_words])
            slope, intercept = np.polyfit(x_positions, y_positions, 1)
            residuals = np.abs(y_positions - (slope * x_positions + intercept))
            if residuals.max() <= img.height * 0.005:
                line_words.sort(key=lambda word: word[1])
            else:
                line_words.sort(key=lambda word: (word[0], word[1]))
        else:
            line_words.sort(key=lambda word: word[1])
        title = " ".join(word[2] for word in line_words)
        # Keep the most confidently read price; None when nothing qualified.
        price = max(price_candidates, default=(0, 0, None))[2]
        return title, price
    except Exception as e:
        console.error(f"OCR failed for {tag_image_path}: {e}")
        return None, None


def extract_product_name(tag_image_path):
    """
    Extract the title text from a close-up out-of-stock tag image.

    Args:
        tag_image_path: Path to the close-up shelf-tag image.

    Returns:
        The recognized product title, or None when no title is found.
    """
    title, _ = extract_tag_details(tag_image_path)
    return title


class TagReader:
    """
    Reads product titles and prices from close-up tag photos.

    Tesseract OCR gives a fast first reading; the local vision model, when
    available, produces a more reliable one.

    Attributes:
        client: Vision-model client used by read_with_vision.
    """

    def __init__(self, client: OllamaClient) -> None:
        """
        Store the vision client.

        Args:
            client: Client for the local vision model.

        Returns:
            None.
        """
        self.client = client

    def read_with_ocr(self, tag_path: str) -> TagInfo:
        """
        Read a tag with Tesseract OCR.

        Args:
            tag_path: Path to the close-up shelf-tag image.

        Returns:
            The OCR title, or UNREADABLE_TITLE when OCR finds none, and the
            price (None when not recognized).
        """
        title, price = extract_tag_details(tag_path)
        return TagInfo(title or UNREADABLE_TITLE, price)

    def read_with_vision(self, tag_path: str) -> Optional[TagInfo]:
        """
        Read a tag with the local vision model.

        Args:
            tag_path: Path to a close-up electronic shelf-tag image.

        Returns:
            The title and price, with price possibly None, or None if the
            model cannot return a usable title.

        Raises:
            RuntimeError: If the vision request fails.
        """
        result = self.client.query(
            "Read this close-up electronic shelf tag. Transcribe the complete "
            "product name exactly as printed, including words on wrapped title "
            "lines. Also read the large current sale price, without the dollar "
            "sign. Ignore size, regular price, savings, barcodes, and "
            'promotional text. Return JSON only: {"title":"...","price":"15.99"}.',
            [tag_path],
        )
        if not isinstance(result, dict):
            return None
        title = result.get("title")
        if not isinstance(title, str):
            return None
        title = " ".join(title.split()).upper()
        if len(title) < 3 or title in {"UNKNOWN", "UNREADABLE", "NONE"}:
            return None

        price_match = re.search(
            r"(\d{1,3})[.,](\d{2})", str(result.get("price", ""))
        )
        price = (
            f"{price_match.group(1)}.{price_match.group(2)}"
            if price_match
            else None
        )
        return TagInfo(title, price)
