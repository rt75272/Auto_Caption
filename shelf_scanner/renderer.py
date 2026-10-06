"""
Render scanner results on copies of their matched shelf-bay images.

Only confirmed empty-slot rectangles are drawn. The output deliberately
matches the provided example: a simple yellow outline around each vacancy,
with no callout panels or connector lines. Original inputs, including the
reference image, are never modified. Bay photos with several missing products
are grouped into a single annotated output image.

Provides:
    caption_bays: group annotations and save marked bay-image copies.
    BayRenderer: renderer object used by the pipeline.

Used by: pipeline.py.
"""

import os

from PIL import Image, ImageDraw
from colorama import Fore, Style
from tqdm import tqdm

from .config import OUTPUT_DIRECTORY_NAME
from .models import Annotation


def caption_bays(annotations, folder_path):
    """
    Highlight verified empty shelf slots in grouped bay images.

    Args:
        annotations: Annotation dictionaries from analyze_and_map_images.
        folder_path: Directory where annotated_bays output is written.

    Returns:
        None. Outputs are saved as annotated copies of the source bay images.
    """
    output_dir = os.path.join(folder_path, OUTPUT_DIRECTORY_NAME)
    os.makedirs(output_dir, exist_ok=True)

    unlocated_count = sum(
        annotation.product_box is None for annotation in annotations
    )
    if unlocated_count:
        tqdm.write(
            f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} Skipping "
            f"{unlocated_count} tag label(s) without a verified empty-slot "
            "location; no label will be drawn in an arbitrary image corner."
        )

    grouped_annotations = {}
    for annotation in annotations:
        if annotation.product_box is None:
            continue
        grouped_annotations.setdefault(annotation.bay_path, []).append(
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
            line_width = max(8, int(min(bay_img.size) * 0.006))
            product_boxes = [
                tuple(int(round(value)) for value in item.product_box)
                for item in bay_annotations
                if item.product_box is not None
            ]

            for product_box in product_boxes:
                draw.rectangle(
                    product_box,
                    outline=(255, 214, 0),
                    width=line_width,
                )

            filename = os.path.basename(bay_path)
            output_path = os.path.join(output_dir, f"annotated_{filename}")
            bay_img.save(output_path)
            product_titles = ", ".join(
                item.title for item in bay_annotations
            )
            tqdm.write(
                f"{Fore.GREEN}[OK]{Style.RESET_ALL} Highlighted "
                f"{len(product_boxes)} empty slot(s) for {product_titles} "
                f"on {output_path}"
            )

        except Exception as e:
            tqdm.write(
                f"{Fore.RED}[ERROR]{Style.RESET_ALL} Could not annotate "
                f"{bay_path}: {e}"
            )

class BayRenderer:
    """Write annotated copies of bays that have confirmed empty slots.

    Args:
        None.

    Returns:
        None.
    """

    def render(self, annotations, folder_path: str) -> None:
        """Group annotations by bay and save outlined image copies.

        Args:
            annotations: Annotation records produced by ShelfScanner.
            folder_path: Root input folder for the scan and output directory.

        Returns:
            None. Annotated copies are written to annotated_bays.
        """
        caption_bays(annotations, folder_path)
