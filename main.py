import base64
import json
import os
import re
from difflib import SequenceMatcher
from datetime import datetime
from io import BytesIO
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from colorama import Fore, Style, just_fix_windows_console
from tqdm import tqdm

just_fix_windows_console()

SCENE_GRID_SIZE = 8
MIN_BAY_GRID_COVERAGE = 40
MAX_BAY_AGE_SECONDS = 180
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
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("AUTOCAPTION_VISION_MODEL", "qwen3.5:latest")


def extract_tag_details(tag_image_path):
    """Read a likely title line and displayed price from a close-up tag."""
    try:
        img = Image.open(tag_image_path)
        data = pytesseract.image_to_data(
            img,
            config="--oem 3 --psm 11",
            output_type=pytesseract.Output.DICT,
        )
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
            if (
                price_match
                and float(data["conf"][index]) >= 30
                and img.height * 0.2 <= int(data["top"][index]) <= img.height * 0.7
            ):
                price = f"{price_match.group(1)}.{price_match.group(2)}"
                price_candidates.append(
                    (confidence, int(data["top"][index]), price)
                )

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
        price = max(price_candidates, default=(0, 0, None))[2]
        return title, price
    except Exception as e:
        tqdm.write(
            f"{Fore.RED}[ERROR]{Style.RESET_ALL} OCR failed for "
            f"{tag_image_path}: {e}"
        )
        return None, None


def extract_product_name(tag_image_path):
    """Extract the title text from the close-up out-of-stock image."""
    title, _ = extract_tag_details(tag_image_path)
    return title


def capture_timestamp(path):
    match = re.search(r"(\d{8})_(\d{9})", os.path.basename(path))
    if match is None:
        return None
    return datetime.strptime(
        match.group(1) + match.group(2)[:6],
        "%Y%m%d%H%M%S",
    )


def read_bay_text(image_path, cache):
    """Read visible text and word locations from a bay image once per image."""
    if image_path not in cache:
        image = Image.open(image_path).convert("RGB")
        data = pytesseract.image_to_data(
            image,
            config="--oem 3 --psm 11",
            output_type=pytesseract.Output.DICT,
        )
        cache[image_path] = [
            (
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


def find_title_box(title, price, bay_words, image_size):
    """Estimate a product-slot rectangle from OCR text or its shelf price."""
    title_tokens = [
        re.sub(r"[^A-Z0-9]", "", token.upper())
        for token in title.split()
    ]
    title_tokens = [
        token for token in title_tokens
        if len(token) >= 4 and token not in TITLE_METADATA_WORDS
    ]
    matches = []
    for token in title_tokens:
        for word in bay_words:
            recognized, confidence, left, top, width, height, _ = word
            if confidence < 25 or len(recognized) < 4:
                continue
            similarity = SequenceMatcher(None, token, recognized).ratio()
            if similarity >= 0.78:
                matches.append(
                    (
                        left + width / 2,
                        top + height / 2,
                        token,
                        confidence * similarity,
                        left,
                        top,
                        left + width,
                        top + height,
                    )
                )

    if matches:
        image_width, image_height = image_size
        radius_x = image_width * 0.12
        radius_y = image_height * 0.08
        best_group = max(
            matches,
            key=lambda match: sum(
                1
                for other in matches
                if abs(other[0] - match[0]) <= radius_x
                and abs(other[1] - match[1]) <= radius_y
            ),
        )
        nearby = [
            match
            for match in matches
            if abs(match[0] - best_group[0]) <= radius_x
            and abs(match[1] - best_group[1]) <= radius_y
        ]
        unique_tokens = {match[2] for match in nearby}
        strongest = max(nearby, key=lambda match: match[3])
        if len(unique_tokens) >= 2 or (
            len(strongest[2]) >= 8 and strongest[3] >= 70
        ):
            text_left = min(match[4] for match in nearby)
            text_top = min(match[5] for match in nearby)
            text_right = max(match[6] for match in nearby)
            text_bottom = max(match[7] for match in nearby)
            center_x = (text_left + text_right) / 2
            center_y = (text_top + text_bottom) / 2
            box_width = min(
                image_width * 0.25,
                max((text_right - text_left) * 3, image_width * 0.08),
            )
            box_height = min(
                image_height * 0.25,
                max((text_bottom - text_top) * 5, image_height * 0.12),
            )
            return (
                max(0, center_x - box_width / 2),
                max(0, center_y - box_height / 2),
                min(image_width, center_x + box_width / 2),
                min(image_height, center_y + box_height / 2),
            )

    if price:
        normalized_price = re.sub(r"\D", "", price)
        price_hits = [
            word
            for word in bay_words
            if re.sub(r"\D", "", word[6]) == normalized_price
            and word[1] >= 35
        ]
        if len(price_hits) == 1:
            _, _, left, top, width, _, _ = price_hits[0]
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


def check_vision_model():
    """Check whether the configured local Ollama vision model is available."""
    try:
        with urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as response:
            models = json.load(response).get("models", [])
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return False

    if any(model.get("name") == OLLAMA_MODEL for model in models):
        return True

    tqdm.write(
        f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} Local vision model "
        f"{OLLAMA_MODEL!r} is not installed. Run "
        f"`ollama pull {OLLAMA_MODEL}` to enable AI product matching."
    )
    return False


def query_vision_model(prompt, image_paths, max_image_size=1024):
    """Send resized local images to Ollama and parse its JSON response."""
    images = []
    for path in image_paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        image.thumbnail((max_image_size, max_image_size))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        images.append(base64.b64encode(buffer.getvalue()).decode("ascii"))

    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": images,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 120},
        }
    ).encode("utf-8")
    request = Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Local vision request to {OLLAMA_URL} failed: {error}"
        ) from error

    model_response = result.get("response", "")
    fenced_response = re.search(
        r"```(?:json)?\s*(.*?)```", model_response, flags=re.DOTALL
    )
    if fenced_response:
        model_response = fenced_response.group(1)
    else:
        json_response = re.search(
            r"(\[[\s\S]*\]|\{[\s\S]*\})", model_response
        )
        if json_response:
            model_response = json_response.group(1)

    try:
        return json.loads(model_response)
    except json.JSONDecodeError:
        return None


def read_tag_with_vision(tag_path):
    """Read the exact product title and current price from a close-up tag."""
    result = query_vision_model(
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

    price_match = re.search(r"(\d{1,3})[.,](\d{2})", str(result.get("price", "")))
    price = (
        f"{price_match.group(1)}.{price_match.group(2)}"
        if price_match
        else None
    )
    return title, price


def locate_with_vision(title, price, bay_path, image_size):
    """Locate a product or its likely shelf slot within a full bay image."""
    price_hint = f"${price}" if price else "not legible"
    result = query_vision_model(
        f'Find the exact package named "{title}", current price '
        f"{price_hint}, in this grocery bay photo. Use the brand and full "
        "product name to distinguish it from similar products. If the "
        "product is present, box the package. If it is out of stock, find "
        "its matching shelf price and box the empty product-facing slot "
        "immediately above that price, using neighboring packages to "
        "estimate the missing product's area. Do not box the price label. "
        "Return JSON only: "
        '{"bbox_2d":[left,top,right,bottom]}, coordinates are 0..1000 '
        "fractions of the full image, x then y from the top-left. If the "
        "exact package cannot be identified, return null.",
        [bay_path],
        max_image_size=1536,
    )
    if isinstance(result, dict):
        if "bbox_2d" in result or "bbox" in result:
            detections = [result]
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
    if not isinstance(detections, list) or not detections:
        return None

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
        scale = 1.0 if max(coordinates) <= 1 else 1000.0
        left, top, right, bottom = (
            value / scale for value in coordinates
        )
        if not (
            0 <= left < right <= 1
            and 0 <= top < bottom <= 1
            and right - left <= 0.6
            and bottom - top <= 0.6
        ):
            continue

        try:
            confidence = float(detection.get("confidence", 0.65))
        except (TypeError, ValueError):
            continue
        if confidence < 0.45:
            continue
        return (
            title,
            (
                left * image_size[0],
                top * image_size[1],
                right * image_size[0],
                bottom * image_size[1],
            ),
        )

    return None


def has_wide_scene_coverage(keypoints, image_shape):
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


def analyze_and_map_images(folder_path):
    """Read each close-up title and map it to a bay image independently."""
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(
            f"{Fore.RED}[ERROR]{Style.RESET_ALL} Tesseract OCR is required but "
            "was not found. Install it with `sudo apt-get install "
            "tesseract-ocr`, then run this script again."
        ) from error
    vision_enabled = check_vision_model()
    if vision_enabled:
        tqdm.write(
            f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Using local vision model "
            f"{OLLAMA_MODEL} for product placement."
        )

    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(all_files) < 2:
        print(
            f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} Not enough images in "
            "folder to process."
        )
        return

    print(
        f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loaded {len(all_files)} images. "
        "Analyzing visual scale and mapping..."
    )

    # Use ORB keypoint coverage to distinguish wide bay scenes.
    orb = cv2.ORB_create(nfeatures=2000)
    img_data = {}
    for path in tqdm(
        all_files, desc="Loading images", unit="image", colour="cyan"
    ):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        kp = orb.detect(img, None)
        img_data[path] = {'shape': img.shape, 'kp': kp}

    macro_focus_paths = {
        path for path in img_data
        if "macro_focus" in os.path.basename(path).lower()
    }
    bay_paths = {
        path
        for path, data in img_data.items()
        if path not in macro_focus_paths
        and has_wide_scene_coverage(data['kp'], data['shape'])
    }

    if not macro_focus_paths:
        tqdm.write(
            f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} No images with "
            "'MACRO_FOCUS' in the filename were found."
        )
        return {}
    if not bay_paths:
        tqdm.write(
            f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} No bay images found "
            "without 'MACRO_FOCUS' in the filename."
        )
        return {}

    tag_paths = [
        path for path in sorted(macro_focus_paths)
    ]
    candidate_bay_paths = sorted(bay_paths)
    bay_text_cache = {}
    annotations = []

    with tqdm(
        total=len(tag_paths),
        desc="Reading tags and locating products",
        unit="tag",
        colour="blue",
    ) as progress:
        for path_a in tag_paths:
            tag_time = capture_timestamp(path_a)
            preceding_bays = [
                path_b
                for path_b in candidate_bay_paths
                if tag_time is not None
                and capture_timestamp(path_b) is not None
                and capture_timestamp(path_b) <= tag_time
                and (tag_time - capture_timestamp(path_b)).total_seconds()
                <= MAX_BAY_AGE_SECONDS
            ]
            if preceding_bays:
                path_b = max(preceding_bays, key=capture_timestamp)
            elif tag_time is not None:
                timed_bays = [
                    path_b for path_b in candidate_bay_paths
                    if capture_timestamp(path_b) is not None
                ]
                path_b = min(
                    timed_bays,
                    key=lambda candidate: abs(
                        (capture_timestamp(candidate) - tag_time).total_seconds()
                    ),
                    default=None,
                )
            else:
                path_b = candidate_bay_paths[0] if candidate_bay_paths else None

            if path_b is None:
                tqdm.write(
                    f"{Fore.RED}[ERROR]{Style.RESET_ALL} No bay image can be "
                    f"paired with {os.path.basename(path_a)}."
                )
                progress.update(1)
                continue

            title = None
            try:
                title, price = extract_tag_details(path_a)
                title = title or "TITLE UNREADABLE"
                bay_image = Image.open(path_b)
                product_box = None
                if vision_enabled:
                    try:
                        vision_title = read_tag_with_vision(path_a)
                        if vision_title:
                            title, price = vision_title
                        if title != "TITLE UNREADABLE":
                            vision_match = locate_with_vision(
                                title,
                                price,
                                path_b,
                                bay_image.size,
                            )
                            if vision_match:
                                title, product_box = vision_match
                    except RuntimeError as error:
                        tqdm.write(
                            f"{Fore.RED}[ERROR]{Style.RESET_ALL} {error}. "
                            "Disabling local vision matching for the "
                            "remaining images."
                        )
                        vision_enabled = False

                if title == "TITLE UNREADABLE":
                    tqdm.write(
                        f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} Could not "
                        f"read a title from {os.path.basename(path_a)}."
                    )

                if product_box is None:
                    product_box = find_title_box(
                        title,
                        price,
                        read_bay_text(path_b, bay_text_cache),
                        bay_image.size,
                    )
                if product_box is None:
                    tqdm.write(
                        f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} No "
                        f"reliable product location found for {title!r} in "
                        f"{os.path.basename(path_b)}."
                    )

                annotations.append(
                    {
                        "bay_path": path_b,
                        "tag_path": path_a,
                        "title": title,
                        "product_box": product_box,
                    }
                )
            except Exception as error:
                tqdm.write(
                    f"{Fore.RED}[ERROR]{Style.RESET_ALL} Could not fully "
                    f"process {os.path.basename(path_a)}: {error}. "
                    "Creating a fallback annotation so the tag is not omitted."
                )
                bay_image = Image.open(path_b)
                annotations.append(
                    {
                        "bay_path": path_b,
                        "tag_path": path_a,
                        "title": title or "TITLE UNREADABLE",
                        "product_box": None,
                    }
                )
            finally:
                progress.update(1)

    return annotations

def caption_bays(annotations, folder_path):
    """Render all tagged product areas together on one image per bay."""
    output_dir = os.path.join(folder_path, "annotated_bays")
    os.makedirs(output_dir, exist_ok=True)

    grouped_annotations = {}
    for annotation in annotations:
        grouped_annotations.setdefault(annotation["bay_path"], []).append(
            annotation
        )

    for annotation in tqdm(
        grouped_annotations.items(),
        desc="Annotating matched bay images",
        unit="bay",
        colour="yellow",
    ):
        bay_path, bay_annotations = annotation
        try:
            bay_img = Image.open(bay_path).convert("RGB")
            draw = ImageDraw.Draw(bay_img)
            font_path_options = (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
                "arial.ttf",
            )

            def load_font(size):
                for font_path in font_path_options:
                    try:
                        return ImageFont.truetype(font_path, size)
                    except OSError:
                        continue
                return ImageFont.load_default(size=size)

            def overlap_area(first, second):
                width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
                height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
                return width * height

            base_font_size = min(
                32, max(16, int(min(bay_img.size) * 0.008))
            )
            max_text_width = int(bay_img.width * 0.32)
            prepared = []
            product_boxes = []

            for item in bay_annotations:
                product_box = item["product_box"]
                if product_box is not None:
                    product_box = tuple(
                        int(round(value)) for value in product_box
                    )
                    product_boxes.append(product_box)

                font_size = base_font_size
                while True:
                    font = load_font(font_size)
                    lines = []
                    current_line = ""
                    for word in item["title"].split():
                        candidate = f"{current_line} {word}".strip()
                        if (
                            current_line
                            and draw.textbbox((0, 0), candidate, font=font)[2]
                            > max_text_width
                        ):
                            lines.append(current_line)
                            current_line = word
                        else:
                            current_line = candidate
                    if current_line:
                        lines.append(current_line)
                    caption_text = "\n".join(lines) or "TITLE UNREADABLE"
                    text_bbox = draw.multiline_textbbox(
                        (0, 0),
                        caption_text,
                        font=font,
                        spacing=font_size // 5,
                    )
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    if (
                        (
                            text_width <= max_text_width
                            and text_height <= bay_img.height * 0.14
                        )
                        or font_size <= 12
                    ):
                        break
                    font_size -= 2

                padding = max(7, font_size // 3)
                prepared.append(
                    {
                        "product_box": product_box,
                        "font": font,
                        "text": caption_text,
                        "spacing": font_size // 5,
                        "padding": padding,
                        "width": text_width + padding * 2,
                        "height": text_height + padding * 2,
                    }
                )

            for product_box in product_boxes:
                draw.rectangle(
                    product_box,
                    outline=(0, 0, 0),
                    width=max(8, base_font_size // 2),
                )
                draw.rectangle(
                    product_box,
                    outline=(0, 255, 230),
                    width=max(4, base_font_size // 4),
                )

            placed_labels = []
            for index, item in enumerate(prepared):
                width, height = item["width"], item["height"]
                product_box = item["product_box"]
                gap = max(8, base_font_size // 2)
                if product_box is None:
                    candidates = [
                        (gap, gap + index * (height + gap)),
                        (
                            bay_img.width - width - gap,
                            gap + index * (height + gap),
                        ),
                    ]
                else:
                    left, top, right, bottom = product_box
                    center_x = (left + right) // 2
                    candidates = [
                        (center_x - width // 2, top - height - gap),
                        (center_x - width // 2, bottom + gap),
                        (left - width - gap, (top + bottom - height) // 2),
                        (right + gap, (top + bottom - height) // 2),
                        (right + gap, top - height - gap),
                        (left - width - gap, top - height - gap),
                        (right + gap, bottom + gap),
                        (left - width - gap, bottom + gap),
                    ]

                best_position = None
                best_score = None
                for candidate_x, candidate_y in candidates:
                    candidate_x = max(
                        0, min(candidate_x, bay_img.width - width)
                    )
                    candidate_y = max(
                        0, min(candidate_y, bay_img.height - height)
                    )
                    candidate_box = (
                        candidate_x,
                        candidate_y,
                        candidate_x + width,
                        candidate_y + height,
                    )
                    collision_cost = sum(
                        overlap_area(candidate_box, placed) * 1000
                        for placed in placed_labels
                    )
                    collision_cost += sum(
                        overlap_area(candidate_box, target) * 100
                        for target in product_boxes
                    )
                    if product_box is None:
                        distance_cost = candidate_x + candidate_y
                    else:
                        target_x = (product_box[0] + product_box[2]) / 2
                        target_y = (product_box[1] + product_box[3]) / 2
                        label_x = (candidate_x + candidate_box[2]) / 2
                        label_y = (candidate_y + candidate_box[3]) / 2
                        distance_cost = abs(label_x - target_x) + abs(
                            label_y - target_y
                        )
                    score = collision_cost + distance_cost
                    if best_score is None or score < best_score:
                        best_score = score
                        best_position = (candidate_x, candidate_y)

                left, top = best_position
                item["label_box"] = (
                    left,
                    top,
                    left + width,
                    top + height,
                )
                placed_labels.append(item["label_box"])

            for item in prepared:
                product_box = item["product_box"]
                if product_box is None:
                    continue
                label_box = item["label_box"]
                target_x = (product_box[0] + product_box[2]) // 2
                target_y = (product_box[1] + product_box[3]) // 2
                label_x = (label_box[0] + label_box[2]) // 2
                label_y = (label_box[1] + label_box[3]) // 2
                draw.line(
                    (target_x, target_y, label_x, label_y),
                    fill=(0, 0, 0),
                    width=max(7, base_font_size // 3),
                )
                draw.line(
                    (target_x, target_y, label_x, label_y),
                    fill=(255, 214, 0),
                    width=max(3, base_font_size // 7),
                )

            for item in prepared:
                left, top, _, _ = item["label_box"]
                padding = item["padding"]
                draw.rounded_rectangle(
                    item["label_box"],
                    radius=padding,
                    fill=(255, 214, 0),
                    outline=(15, 35, 65),
                    width=max(3, base_font_size // 8),
                )
                draw.multiline_text(
                    (left + padding, top + padding),
                    item["text"],
                    font=item["font"],
                    fill=(15, 35, 65),
                    spacing=item["spacing"],
                )

            filename = os.path.basename(bay_path)
            output_path = os.path.join(output_dir, f"annotated_{filename}")
            bay_img.save(output_path)
            tqdm.write(
                f"{Fore.GREEN}[OK]{Style.RESET_ALL} Annotated "
                f"{len(bay_annotations)} tag(s) on {output_path}"
            )

        except Exception as e:
            tqdm.write(
                f"{Fore.RED}[ERROR]{Style.RESET_ALL} Could not annotate "
                f"{bay_path}: {e}"
            )

if __name__ == "__main__":
    # Provide the directory holding both full-size bays and up-close tags
    target_folder = "./10-06-2026" 
    annotations = analyze_and_map_images(target_folder)
    
    if annotations:
        print(
            f"{Fore.GREEN}[OK]{Style.RESET_ALL} Read "
            f"{len(annotations)} close-up tag(s)."
        )
        caption_bays(annotations, target_folder)
    else:
        print(
            f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} No matching bay "
            "images or products were found."
        )