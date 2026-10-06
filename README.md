 # Auto Caption
Auto Caption is an image-captioning project: it accepts an image and generates a short text description of its contents. The exact model, supported formats, and output options depend on the implementation included in this repository.
## Setup
Use Python 3. From the repository root, create a virtual environment and install the dependencies:
```bash
uv sync --all-groups
source .venv/bin/activate
```
## Run
Run the project's application or script from the repository root. For a project with a `main.py` entry point, for example:
```bash
python main.py
```
Follow the prompts or command-line options to provide an image. Check the entry-point script or its `--help` output for the exact invocation and options supported by this checkout.
## Workflow
1. Provide an image to the application.
2. The configured captioning model analyzes the image.
3. Read or use the generated caption as text.
