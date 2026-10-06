"""
End-to-end shelf scan orchestration.

ShelfScanner coordinates the catalog, tag reader, bay OCR cache, slot
locator, and renderer. It discovers photos below the configured input folder,
pairs each MACRO_FOCUS tag to the latest preceding wide bay photo, then
records only visibly confirmed empty product areas. OCR remains available
when Ollama is not installed or temporarily unreachable; uncertain slot
locations are logged and left unannotated rather than guessed.

Provides:
    ShelfScanner.run: execute a scan and render confirmed bay annotations.
    ShelfScanner.analyze: produce annotation records without rendering them.

Used by: main.py.
"""

import os
from typing import List, Optional

import pytesseract
from PIL import Image
from colorama import Fore, Style
from tqdm import tqdm

from . import console
from .bay_text import BayTextReader, find_empty_slot_box
from .catalog import ImageCatalog
from .config import OLLAMA_MODEL, UNREADABLE_TITLE
from .models import Annotation
from .ollama_client import OllamaClient
from .placement import ShelfSlotLocator
from .renderer import BayRenderer
from .tag_reader import TagReader


class ShelfScanner:
    """Coordinate image discovery, tag reading, slot detection, and output.

    Attributes:
        client: Local Ollama client shared by OCR/vision reader and locator.
        catalog: Recursive source-image catalog and bay/tag matcher.
        tag_reader: Reads product titles and prices from close-up tags.
        bay_text_reader: OCR cache for bay photos.
        slot_locator: Validates candidate empty shelf areas.
        renderer: Writes highlighted copies of matched bay photos.
    """

    def __init__(self, client: Optional[OllamaClient] = None) -> None:
        """Create one set of scan services and share their vision client.

        Args:
            client: Optional preconfigured Ollama client, useful for tests or
                callers that want to customize the server/model.

        Returns:
            None.
        """
        self.client = client or OllamaClient()
        self.catalog = ImageCatalog()
        self.tag_reader = TagReader(self.client)
        self.bay_text_reader = BayTextReader()
        self.slot_locator = ShelfSlotLocator(self.client)
        self.renderer = BayRenderer()

    def run(self, folder_path: str) -> List[Annotation]:
        """Analyze source photos and save annotated copies for confirmed slots.

        Args:
            folder_path: Root directory that contains bay and tag images.

        Returns:
            One Annotation record per successfully paired tag image.
        """
        annotations = self.analyze(folder_path)
        if annotations:
            console.ok(f"Read {len(annotations)} close-up tag(s).")
            self.renderer.render(annotations, folder_path)
        else:
            console.warning("No matching bay images or products were found.")
        return annotations

    def analyze(self, folder_path: str) -> List[Annotation]:
        """Pair close-up tags with bays and locate any confirmed empty slots.

        Args:
            folder_path: Root directory containing the scan's source images.

        Returns:
            Annotation records for tags that could be paired with a bay.
            Unconfirmed product locations have product_box=None.

        Raises:
            RuntimeError: If the required Tesseract OCR executable is missing.
            FileNotFoundError: If the source folder does not exist.
        """
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError as error:
            raise RuntimeError(
                f"{Fore.RED}[ERROR]{Style.RESET_ALL} Tesseract OCR is "
                "required but was not found. Install it with `sudo apt-get "
                "install tesseract-ocr`, then run this script again."
            ) from error

        groups = self.catalog.discover(folder_path)
        if len(groups.all_files) < 2:
            console.warning("Not enough images in folder to process.")
            return []

        console.info(
            f"Loaded {len(groups.all_files)} images. Analyzing visual scale "
            "and mapping..."
        )
        if not groups.tag_paths:
            console.warning(
                "No images with 'MACRO_FOCUS' in the filename were found."
            )
            return []
        if not groups.bay_paths:
            console.warning(
                "No bay images found without 'MACRO_FOCUS' in the filename."
            )
            return []

        vision_enabled = self.client.is_ready()
        if vision_enabled:
            console.info(
                f"Using local vision model {OLLAMA_MODEL} for product "
                "placement."
            )

        annotations: List[Annotation] = []
        with tqdm(
            total=len(groups.tag_paths),
            desc="Reading tags and locating products",
            unit="tag",
            colour="blue",
        ) as progress:
            for tag_path in groups.tag_paths:
                bay_path = self.catalog.find_matching_bay(
                    tag_path,
                    groups.bay_paths,
                )
                if bay_path is None:
                    console.error(
                        f"No bay image can be paired with "
                        f"{os.path.basename(tag_path)}."
                    )
                    progress.update(1)
                    continue

                title = UNREADABLE_TITLE
                price = None
                try:
                    tag_info = self.tag_reader.read_with_ocr(tag_path)
                    title = tag_info.title or UNREADABLE_TITLE
                    price = tag_info.price
                    with Image.open(bay_path) as bay_image:
                        image_size = bay_image.size
                    bay_words = self.bay_text_reader.read(bay_path)
                    product_box = None

                    if vision_enabled:
                        try:
                            vision_info = self.tag_reader.read_with_vision(
                                tag_path
                            )
                            if vision_info is not None:
                                title = vision_info.title or title
                                price = vision_info.price or price
                            if title != UNREADABLE_TITLE:
                                vision_match = self.slot_locator.locate(
                                    title,
                                    price,
                                    bay_path,
                                    image_size,
                                    bay_words=bay_words,
                                )
                                if vision_match is not None:
                                    title, product_box = vision_match
                        except RuntimeError as error:
                            console.error(
                                f"{error}. Disabling local vision matching "
                                "for the remaining images."
                            )
                            vision_enabled = False

                    if title == UNREADABLE_TITLE:
                        console.warning(
                            f"Could not read a title from "
                            f"{os.path.basename(tag_path)}."
                        )

                    if product_box is None:
                        # Price OCR gives only a candidate; visual verification
                        # is still mandatory before drawing any rectangle.
                        candidate_box = find_empty_slot_box(
                            price,
                            bay_words,
                            image_size,
                        )
                        if candidate_box is not None and vision_enabled:
                            try:
                                if self.slot_locator.verify(
                                    candidate_box,
                                    bay_path,
                                    image_size,
                                ):
                                    product_box = candidate_box
                            except RuntimeError as error:
                                console.error(
                                    f"{error}. Disabling local vision "
                                    "checks for the remaining images."
                                )
                                vision_enabled = False

                    if product_box is None:
                        console.warning(
                            f"No visibly empty shelf slot confirmed for "
                            f"{title!r} in {os.path.basename(bay_path)}; "
                            "leaving it unannotated rather than marking "
                            "stocked goods."
                        )
                    annotations.append(
                        Annotation(
                            bay_path=bay_path,
                            tag_path=tag_path,
                            title=title,
                            product_box=product_box,
                        )
                    )
                except Exception as error:
                    console.error(
                        f"Could not fully process {os.path.basename(tag_path)}: "
                        f"{error}. Creating a fallback annotation so the tag "
                        "is not omitted."
                    )
                    annotations.append(
                        Annotation(
                            bay_path=bay_path,
                            tag_path=tag_path,
                            title=title,
                            product_box=None,
                        )
                    )
                finally:
                    progress.update(1)
        return annotations
