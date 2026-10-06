"""
Vision-based empty-slot detection with OCR and independent vacancy checks.

Candidate boxes returned by the local vision model are normalized against the
full bay image, screened for readable package text, and checked against the
matching shelf-price area when OCR can uniquely identify it. Every accepted
box must then pass a second vision request focused on the marked merchandise
area. When the first candidate is rejected, a contextual crop may be used to
refine the location. Uncertain or malformed results fail closed.

Provides:
    verify_empty_slot_with_vision: independently confirm a candidate vacancy.
    refine_empty_slot_near_candidate: refine an uncertain bay/candidate crop.
    locate_price_label_with_vision: find a matching shelf-label anchor.
    locate_with_vision: detect, validate, and return an empty product area.
    ShelfSlotLocator: bind those functions to one OllamaClient.

Used by: pipeline.py.
"""

import re
from typing import Optional, Tuple

from PIL import Image, ImageDraw

from .config import TITLE_METADATA_WORDS
from .geometry import Box
from .models import OcrWord
from .ollama_client import OllamaClient
from .bay_text import box_contains_product_text, find_price_label_word

DEFAULT_CLIENT = OllamaClient()


def verify_empty_slot_with_vision(
    product_box, bay_path, image_size, client=None
):
    """
    Confirm that a proposed rectangle is an empty product-facing shelf area.

    The model receives a magnified crop with the candidate outlined, so it
    can distinguish merchandise space from a rail or price-tag holder.
    Missing, malformed, or uncertain answers fail closed.

    Args:
        product_box: Candidate pixel rectangle (left, top, right, bottom).
        bay_path: Path to the original bay image.
        image_size: Original bay dimensions as (width, height), in pixels.

    Returns:
        True only when the model explicitly confirms the outlined area is
        visibly empty with confidence of at least 0.7.
    """
    client = client or DEFAULT_CLIENT
    left, top, right, bottom = (
        int(round(value)) for value in product_box
    )
    left = max(0, min(left, image_size[0]))
    right = max(0, min(right, image_size[0]))
    top = max(0, min(top, image_size[1]))
    bottom = max(0, min(bottom, image_size[1]))
    if left >= right or top >= bottom:
        return False

    margin_x = max(right - left, int(image_size[0] * 0.04))
    margin_y = max(bottom - top, int(image_size[1] * 0.04))
    crop_left = max(0, left - margin_x)
    crop_top = max(0, top - margin_y)
    crop_right = min(image_size[0], right + margin_x)
    crop_bottom = min(image_size[1], bottom + margin_y)
    with Image.open(bay_path) as source:
        marked_bay = source.convert("RGB").crop(
            (crop_left, crop_top, crop_right, crop_bottom)
        )
    crop_box = (
        left - crop_left,
        top - crop_top,
        right - crop_left,
        bottom - crop_top,
    )
    marker = ImageDraw.Draw(marked_bay)
    marker_width = max(8, int(min(marked_bay.size) * 0.015))
    marker.rectangle(
        crop_box,
        outline=(0, 0, 0),
        width=marker_width + 4,
    )
    marker.rectangle(
        crop_box,
        outline=(255, 0, 0),
        width=marker_width,
    )

    result = client.query(
        "This is a close crop from a grocery shelf. The red rectangle marks "
        "the proposed missing-product area. Is the space inside the red "
        "rectangle a package-sized vacant area on the merchandise-facing "
        "shelf or pegboard? Depending on the shelf layout, products may sit "
        "above or hang below a price label or rail. The label holder, rail, "
        "wall, floor, and space outside the merchandise area are NOT product "
        "slots. Any package inside the box means false. If uncertain, answer "
        "false. Return JSON only: "
        '{"empty_slot":true,"confidence":0.95} or '
        '{"empty_slot":false,"confidence":0.95}.',
        [marked_bay],
        max_image_size=2048,
    )
    if not isinstance(result, dict) or result.get("empty_slot") is not True:
        return False
    try:
        confidence = float(result.get("confidence", 0))
    except (TypeError, ValueError):
        return False
    return confidence >= 0.7

def refine_empty_slot_near_candidate(
    title,
    price,
    candidate_box,
    bay_path,
    image_size,
    bay_words,
    client=None,
):
    """
    Refine a rejected full-bay detection using a contextual shelf crop.

    Args:
        title: Product name read from its close-up shelf tag.
        price: Current price read from the tag, or None if unavailable.
        candidate_box: Initial pixel rectangle around a possible slot or tag.
        bay_path: Path to the wide image that will be annotated.
        image_size: Bay image dimensions as (width, height), in pixels.
        bay_words: OCR word records used to reject occupied product areas.

    Returns:
        A verified pixel rectangle for a visible empty slot, or None if the
        contextual model or vacancy verifier cannot confirm one.
    """
    client = client or DEFAULT_CLIENT
    left, top, right, bottom = candidate_box
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    crop_width = int(min(
        image_size[0],
        max((right - left) * 5, image_size[0] * 0.28),
    ))
    crop_height = int(min(
        image_size[1],
        max((bottom - top) * 8, image_size[1] * 0.34),
    ))
    crop_left = min(
        max(0, int(center_x - crop_width / 2)),
        image_size[0] - crop_width,
    )
    crop_top = min(
        max(0, int(center_y - crop_height / 2)),
        image_size[1] - crop_height,
    )
    crop_right = crop_left + crop_width
    crop_bottom = crop_top + crop_height

    with Image.open(bay_path) as source:
        context = source.convert("RGB").crop(
            (crop_left, crop_top, crop_right, crop_bottom)
        )

    price_hint = f"${price}" if price else "not legible"
    result = client.query(
        f"Inspect this grocery shelf close-up for {title!r} at {price_hint}. "
        "The initial location hint may be the price label itself, not the "
        "product location. Find the empty merchandise-facing shelf or "
        "pegboard space paired with that label, using adjacent stocked items "
        "and the shelf structure to determine whether the product space is "
        "above or below the label. Return a box around a package-sized "
        "vacancy only, not the price label or holder, shelf rail, stocked "
        "products, wall, or floor. If no clearly empty merchandise space is "
        "visible, return "
        '{"status":"no_empty_slot","bbox_2d":null}. Otherwise return JSON '
        "only: "
        '{"status":"empty_slot_found","evidence":"visible empty '
        'merchandise slot","bbox_2d":[left,top,right,bottom]}. Coordinates '
        "are 0..1000 fractions of this crop, x then y from its top-left.",
        [context],
        max_image_size=2048,
    )
    if not isinstance(result, dict) or result.get("status") == "no_empty_slot":
        return None
    box = result.get("bbox_2d") or result.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        coordinates = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if max(coordinates) <= 1:
        normalized = coordinates
    else:
        normalized = [value / 1000 for value in coordinates]
    if any(value < 0 or value > 1 for value in normalized):
        return None

    crop_width = context.width
    crop_height = context.height
    slot_box = (
        crop_left + normalized[0] * crop_width,
        crop_top + normalized[1] * crop_height,
        crop_left + normalized[2] * crop_width,
        crop_top + normalized[3] * crop_height,
    )
    slot_left, slot_top, slot_right, slot_bottom = slot_box
    if (
        slot_left >= slot_right
        or slot_top >= slot_bottom
        or slot_right - slot_left < image_size[0] * 0.025
        or slot_bottom - slot_top < image_size[1] * 0.05
        or slot_right - slot_left > image_size[0] * 0.45
        or slot_bottom - slot_top > image_size[1] * 0.45
        or box_contains_product_text(slot_box, bay_words)
        or not verify_empty_slot_with_vision(
            slot_box, bay_path, image_size, client
        )
    ):
        return None
    return slot_box

def locate_price_label_with_vision(
    title, price, bay_path, image_size, client=None
):
    """
    Find the matching shelf-label anchor when bay OCR cannot locate it.

    Args:
        title: Product name read from its close-up shelf tag.
        price: Current price read from the tag, or None if unavailable.
        bay_path: Path to the wide bay image.
        image_size: Bay image dimensions as (width, height), in pixels.

    Returns:
        A pixel rectangle around the matching shelf label, or None if the
        model cannot identify one confidently.
    """
    client = client or DEFAULT_CLIENT
    price_hint = f" at ${price}" if price else ""
    result = client.query(
        f"Find the electronic shelf label for {title!r}{price_hint} in this "
        "grocery bay image. Return a tight rectangle around the shelf label "
        "itself, not the product or an empty area. The next step will inspect "
        "the nearby merchandise space. If you cannot identify the matching "
        "label, return "
        '{"status":"not_found","bbox_2d":null}. Otherwise return JSON only: '
        '{"status":"label_found","bbox_2d":[left,top,right,bottom]}. '
        "Coordinates are 0..1000 fractions of the full image, x then y from "
        "the top-left.",
        [bay_path],
        max_image_size=2048,
    )
    if not isinstance(result, dict) or result.get("status") == "not_found":
        return None

    box = result.get("bbox_2d") or result.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        coordinates = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    normalized = (
        coordinates
        if max(coordinates) <= 1
        else [value / 1000 for value in coordinates]
    )
    if any(value < 0 or value > 1 for value in normalized):
        return None

    left, top, right, bottom = normalized
    if (
        left >= right
        or top >= bottom
        or right - left > 0.2
        or bottom - top > 0.12
    ):
        return None
    return (
        left * image_size[0],
        top * image_size[1],
        right * image_size[0],
        bottom * image_size[1],
    )

def locate_with_vision(
    title,
    price,
    bay_path,
    image_size,
    bay_words=None,
    client=None,
):
    """
    Find only a visibly empty shelf slot associated with an out-of-stock tag.

    Args:
        title: Product name read from its close-up shelf tag.
        price: Current price read from the tag, or None if unavailable.
        bay_path: Path to the wide image that will be annotated.
        image_size: Bay image dimensions as (width, height), in pixels.
        bay_words: Optional OCR word records for verifying alignment to a
            unique matching shelf-price label.

    Returns:
        A (title, product_box) tuple for a visually empty slot, or None when
        the model cannot establish a visible empty slot.
    """
    client = client or DEFAULT_CLIENT
    price_hint = f"${price}" if price else "not legible"
    # Keep the bay as the only image in this request so all coordinates have
    # one unambiguous reference frame. The close-up is already read as text.
    result = client.query(
        f"In this grocery bay image, find the out-of-stock product "
        f"{title!r} at {price_hint}. Locate its matching shelf price and "
        "inspect the merchandise-facing product area associated with that "
        "label. Depending on the shelf layout, products may sit above or "
        "hang below a price label or rail. Return a box ONLY around an empty, "
        "package-sized gap in the actual merchandise area, excluding the "
        "label, holder, rail, stocked goods, wall, and floor. Estimate a "
        "realistic package footprint from adjacent stock and shelf "
        "boundaries. Never box an in-stock package. If the product appears "
        "to be in stock, the matching price cannot be found, or the gap is "
        "obscured/uncertain, return "
        '{"status":"no_empty_slot","bbox_2d":null}. Otherwise return JSON '
        "only in this exact form: "
        '{"status":"empty_slot_found","evidence":"visible vacant '
        'merchandise space paired with matching price","bbox_2d":[left,top,right,'
        "bottom]}. Coordinates are 0..1000 fractions of the full bay image, "
        "x then y from the top-left.",
        [bay_path],
        max_image_size=2048,
    )
    if isinstance(result, dict):
        if (
            ("bbox_2d" in result or "bbox" in result)
            and "status" not in result
            and isinstance(result.get("empty_slot_found"), str)
        ):
            detections = [
                {
                    **result,
                    "status": "empty_slot_found",
                    "evidence": result["empty_slot_found"],
                }
            ]
        elif "bbox_2d" in result or "bbox" in result:
            detections = [
                {
                    **result,
                    "status": "empty_slot_found",
                    "evidence": result.get(
                        "evidence",
                        "unverified empty-slot candidate; requires visual check",
                    ),
                }
            ]
        elif "empty_slot_found" in result:
            found = result["empty_slot_found"]
            if isinstance(found, dict):
                detections = [found]
            elif isinstance(found, list):
                detections = [
                    detection
                    for detection in found
                    if isinstance(detection, dict)
                ]
            elif isinstance(found, str):
                detections = [
                    {
                        "status": "empty_slot_found",
                        "evidence": found,
                        "bbox_2d": result.get("bbox_2d"),
                        "confidence": result.get("confidence", 0.65),
                    }
                ]
            else:
                detections = []
            for detection in detections:
                detection.setdefault("status", "empty_slot_found")
        else:
            detections = [
                {"label": label, "bbox_2d": box[0]}
                for label, box in result.items()
                if isinstance(label, str)
                and isinstance(box, list)
                and len(box) == 1
                and isinstance(box[0], list)
            ]
    else:
        detections = result

    price_word = find_price_label_word(price, bay_words or [])
    refinement_anchor = None
    if price_word is not None:
        _, _, price_left, price_top, price_width, price_height, _ = price_word
        refinement_anchor = (
            price_left,
            price_top,
            price_left + price_width,
            price_top + price_height,
        )
    if not isinstance(detections, list):
        detections = []

    title_tokens = {
        re.sub(r"[^A-Z0-9]", "", token.upper())
        for token in title.split()
        if len(re.sub(r"[^A-Z0-9]", "", token.upper())) >= 4
        and re.sub(r"[^A-Z0-9]", "", token.upper())
        not in TITLE_METADATA_WORDS
    } if title and title != "TITLE UNREADABLE" else set()

    for detection in detections:
        if not isinstance(detection, dict):
            continue
        status = str(detection.get("status", "")).strip().lower()
        if status not in {"empty_slot_found", "candidate"}:
            continue
        evidence = detection.get("evidence", "")
        if not isinstance(evidence, str) or (
            status != "candidate"
            and not re.search(
                r"\b(empty|vacant|unoccupied|bare|gap)\b",
                evidence,
                flags=re.IGNORECASE,
            )
        ):
            continue
        label = detection.get("label") or detection.get("title") or ""
        box = detection.get("bbox_2d") or detection.get("bbox")
        if not isinstance(label, str) or not isinstance(box, list) or len(box) != 4:
            continue
        label_tokens = {
            re.sub(r"[^A-Z0-9]", "", token.upper())
            for token in label.split()
            if len(re.sub(r"[^A-Z0-9]", "", token.upper())) >= 4
            and re.sub(r"[^A-Z0-9]", "", token.upper())
            not in TITLE_METADATA_WORDS
        }
        if label and not label_tokens:
            continue
        if title_tokens and label_tokens:
            overlap = len(title_tokens & label_tokens) / min(
                len(title_tokens), len(label_tokens)
            )
            if overlap < 0.4:
                continue

        try:
            coordinates = [float(value) for value in box]
        except (TypeError, ValueError):
            continue
        if max(coordinates) <= 1:
            normalized_coordinates = coordinates
        else:
            normalized_coordinates = [
                value / 1000 for value in coordinates
            ]
        if any(
            value < -0.05 or value > 1.05
            for value in normalized_coordinates
        ):
            continue
        left, top, right, bottom = (
            max(0.0, min(1.0, value))
            for value in normalized_coordinates
        )
        if not (
            0 <= left < right <= 1
            and 0 <= top < bottom <= 1
            and right - left <= 0.45
            and bottom - top <= 0.45
        ):
            continue

        try:
            confidence = float(detection.get("confidence", 0.65))
        except (TypeError, ValueError):
            continue
        if confidence < 0.45:
            continue

        # Boxes that fit only a price display or shelf edge are too short to
        # represent the missing package's merchandise-facing footprint.
        if right - left < 0.025 or bottom - top < 0.05:
            refinement_anchor = (
                left * image_size[0],
                top * image_size[1],
                right * image_size[0],
                bottom * image_size[1],
            )
            continue

        # When OCR can uniquely read the shelf price, require a nearby,
        # horizontally aligned box on either side of the label.
        if price_word is not None:
            (
                _,
                _,
                price_left,
                price_top,
                price_width,
                price_height,
                _,
            ) = price_word
            box_left = left * image_size[0]
            box_top = top * image_size[1]
            box_right = right * image_size[0]
            box_bottom = bottom * image_size[1]
            box_center_x = (box_left + box_right) / 2
            price_center_x = price_left + price_width / 2
            if abs(box_center_x - price_center_x) > max(
                price_width * 2,
                image_size[0] * 0.06,
            ):
                continue

            price_bottom = price_top + price_height
            overlaps_price = box_top < price_bottom and box_bottom > price_top
            if overlaps_price:
                refinement_anchor = (
                    box_left,
                    box_top,
                    box_right,
                    box_bottom,
                )
                continue
            vertical_gap = (
                price_top - box_bottom
                if box_bottom < price_top
                else box_top - price_bottom
            )
            if vertical_gap > image_size[1] * 0.22:
                continue

        product_box = (
            left * image_size[0],
            top * image_size[1],
            right * image_size[0],
            bottom * image_size[1],
        )
        if bay_words is not None and box_contains_product_text(
            product_box,
            bay_words,
        ):
            refinement_anchor = product_box
            continue
        if not verify_empty_slot_with_vision(
            product_box,
            bay_path,
            image_size,
            client,
        ):
            refinement_anchor = product_box
            continue
        return title, product_box

    if refinement_anchor is None:
        refinement_anchor = locate_price_label_with_vision(
            title,
            price,
            bay_path,
            image_size,
            client,
        )
    if refinement_anchor is not None:
        refined_box = refine_empty_slot_near_candidate(
            title,
            price,
            refinement_anchor,
            bay_path,
            image_size,
            bay_words or [],
            client,
        )
        if refined_box is not None:
            return title, refined_box
    return None

class ShelfSlotLocator:
    """Locate only visually confirmed empty shelf product spaces.

    Attributes:
        client: Ollama client used for candidate detection and verification.
    """

    def __init__(self, client: OllamaClient) -> None:
        """Bind a locator to the scan's local vision client.

        Args:
            client: Client used to query the configured vision model.

        Returns:
            None.
        """
        self.client = client

    def locate(
        self,
        title: str,
        price: Optional[str],
        bay_path: str,
        image_size: Tuple[int, int],
        bay_words=None,
    ):
        """Find a verified empty slot associated with a tagged product.

        Args:
            title: Product title read from the close-up shelf tag.
            price: Sale price without a currency symbol, if readable.
            bay_path: Path to the wide shelf-bay image.
            image_size: Bay image dimensions as (width, height), in pixels.
            bay_words: OCR words used for occupancy and price checks.

        Returns:
            A (title, pixel box) pair when a slot is confirmed; otherwise
            None.
        """
        result = locate_with_vision(
            title,
            price,
            bay_path,
            image_size,
            bay_words=bay_words,
            client=self.client,
        )
        if result is None:
            return None
        product_title, product_box = result
        return product_title, Box(*product_box)

    def verify(
        self,
        product_box,
        bay_path: str,
        image_size: Tuple[int, int],
    ) -> bool:
        """Run the independent visual vacancy check for a candidate box.

        Args:
            product_box: Candidate pixel rectangle (left, top, right, bottom).
            bay_path: Path to the wide shelf-bay image.
            image_size: Bay image dimensions as (width, height), in pixels.

        Returns:
            True only when the model confirms an empty merchandise area.
        """
        return verify_empty_slot_with_vision(
            product_box,
            bay_path,
            image_size,
            client=self.client,
        )
