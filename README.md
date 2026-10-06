# Shelf Scanner

Shelf Scanner reads product names from close-up electronic shelf-tag photos,
associates each tag with a nearby wide bay photo, and outlines only a
visibly-confirmed empty product space. It uses Tesseract for OCR and an
optional local Ollama vision model for tag reading and slot verification.

## Setup

Install the system OCR executable and the Python dependencies:

```sh
sudo apt-get install tesseract-ocr
uv sync --all-groups
source .venv/bin/activate
```

The local vision model defaults to `qwen3.5:latest`. Pull it once with
`ollama pull qwen3.5:latest` and start Ollama to enable vision matching.
Without a reachable model, the program can still read tags with OCR, but it
will not draw unverified empty-slot boxes.

## Input and output

Put bay images and close-up tag images anywhere below `pictures/`. The
scanner searches subfolders recursively. Close-up tag image names must
include `MACRO_FOCUS`; other images are considered bay candidates when their
visual features indicate a wide shelf scene. Tags are paired with the most
recent bay photo taken before the tag (within the configured time window).

Run the application from the repository root:

```sh
python main.py
```

An alternate root folder can be supplied as a positional argument:

```sh
python main.py /path/to/shelf-images
```

The supplied `example.png` is a visual reference, not input: any image whose
filename stem is `example` is skipped before decoding. Source images are never
modified. Confirmed empty spaces are outlined in yellow on copies written to
`pictures/annotated_bays/` (or the equivalent output folder under the chosen
input root). Uncertain or occupied areas are left unannotated.

To select a different local model or server URL, set environment variables:

```sh
AUTOCAPTION_VISION_MODEL=qwen3.5:latest \
OLLAMA_HOST=http://127.0.0.1:11434 python main.py
```

## Program structure

```text
.
├── main.py                       # Small command-line driver
├── shelf_scanner/
│   ├── __init__.py                # Package metadata
│   ├── config.py                  # Shared settings and environment overrides
│   ├── console.py                 # Colored, progress-bar-safe messages
│   ├── geometry.py                # Bounding boxes and model coordinates
│   ├── models.py                  # OCR, tag, and annotation records
│   ├── catalog.py                 # Recursive discovery and bay/tag pairing
│   ├── bay_text.py                # Bay-image OCR and price/occupancy checks
│   ├── tag_reader.py              # Close-up tag title and price OCR/vision
│   ├── ollama_client.py           # Local Ollama HTTP and image encoding
│   ├── placement.py               # Empty-slot proposal and verification
│   ├── renderer.py                # Grouped yellow-outline output images
│   └── pipeline.py                # ShelfScanner workflow orchestration
└── pictures/                      # Ignored local input images
    └── annotated_bays/            # Generated annotated copies
    └── example/                   # Example images
    └── training/                  # Training images 
```

### Processing flow

1. `main.py` creates `ShelfScanner` and selects the input folder.
2. `catalog.py` recursively discovers images, excludes the reference and old
   outputs, classifies wide bays and MACRO_FOCUS tags, and pairs each tag.
3. `tag_reader.py` reads the tag; `bay_text.py` caches text from the bay.
4. `placement.py` checks candidate empty areas against OCR occupancy and an
   independent visual vacancy confirmation from `ollama_client.py`.
5. `renderer.py` groups confirmed locations by bay and writes annotated
   copies. `models.py` carries typed data between the stages.

## Configuration

Shared constants, including bay-pairing limits and output folder names, live
in `shelf_scanner/config.py`. The vision server URL is controlled by
`OLLAMA_HOST`; the model is controlled by `AUTOCAPTION_VISION_MODEL`.
