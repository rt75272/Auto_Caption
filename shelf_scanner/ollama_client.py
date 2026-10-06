"""
Client for the configured local Ollama vision model.

All model communication and image encoding are kept here rather than spread
across tag reading and placement. A bounded retry handles brief local-server
connection drops while the model is being loaded; if those retries fail, the
caller receives an explicit RuntimeError. Requests ask Ollama to keep the
model warm between successive shelf images.

Provides:
    encode_image: convert a path or Pillow image into base64 JPEG.
    extract_json: decode a JSON value from plain or fenced model output.
    OllamaClient: check model availability and issue vision requests.

Used by: tag_reader.py, placement.py, and pipeline.py.
"""

import base64
import json
import re
import time
from http.client import HTTPException
from io import BytesIO
from typing import Any, List, Optional, Sequence, Tuple, Union
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image

from . import console
from .config import OLLAMA_MODEL, OLLAMA_URL

ImageInput = Union[str, Image.Image]
DEFAULT_RETRY_DELAYS = (2.0, 5.0)


def encode_image(image: ImageInput, max_image_size: int) -> str:
    """Encode one image as a resized, base64 JPEG for the Ollama API.

    Args:
        image: File path or Pillow image to encode.
        max_image_size: Maximum width or height after downscaling.

    Returns:
        Base64-encoded JPEG content.
    """
    if isinstance(image, Image.Image):
        prepared = image.convert("RGB")
    else:
        with Image.open(image) as source:
            prepared = source.convert("RGB")
    prepared.thumbnail((max_image_size, max_image_size))
    buffer = BytesIO()
    prepared.save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def extract_json(model_response: str) -> Any:
    """Parse the first JSON object or array in a model reply.

    Args:
        model_response: Raw text returned by Ollama.

    Returns:
        The parsed JSON value, or None when the reply contains invalid JSON.
    """
    fenced = re.search(
        r"```(?:json)?\s*(.*?)```", model_response, flags=re.DOTALL
    )
    if fenced:
        model_response = fenced.group(1)
    else:
        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", model_response)
        if match:
            model_response = match.group(1)
    try:
        return json.loads(model_response)
    except json.JSONDecodeError:
        return None


class OllamaClient:
    """Send bounded, JSON-oriented image requests to a local Ollama server.

    Attributes:
        url: Base URL of the Ollama server.
        model: Configured vision model name.
        timeout: Maximum seconds allowed per generation request.
        retry_delays: Delay before each bounded request retry.
    """

    def __init__(
        self,
        url: str = OLLAMA_URL,
        model: str = OLLAMA_MODEL,
        timeout: float = 120.0,
        retry_delays: Tuple[float, ...] = DEFAULT_RETRY_DELAYS,
    ) -> None:
        """Configure the local model client without making a network call.

        Args:
            url: Base URL of the Ollama server.
            model: Vision model name to request.
            timeout: Per-request timeout, including model loading.
            retry_delays: Finite delays for retries after a connection failure.

        Returns:
            None.
        """
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retry_delays = retry_delays

    def is_ready(self) -> bool:
        """Check the server and confirm that it lists the configured model.

        Args:
            None.

        Returns:
            True when Ollama lists the configured model; otherwise False and
            a warning that explains whether the server or model is missing.
        """
        try:
            with urlopen(f"{self.url}/api/tags", timeout=3) as response:
                models = json.load(response).get("models", [])
        except (OSError, TimeoutError, json.JSONDecodeError) as error:
            console.warning(
                f"Could not reach the local Ollama server at {self.url}: "
                f"{error}. Continuing with OCR-only tag reading."
            )
            return False
        if any(model.get("name") == self.model for model in models):
            return True
        console.warning(
            f"Local vision model {self.model!r} is not installed. Run "
            f"`ollama pull {self.model}` to enable AI product matching."
        )
        return False

    def query(
        self,
        prompt: str,
        images: Sequence[ImageInput],
        max_image_size: int = 1024,
    ) -> Any:
        """Send an image prompt and return the model's parsed JSON result.

        Args:
            prompt: Instruction for the vision model.
            images: Image paths or Pillow images to include in the request.
            max_image_size: Maximum width or height of each encoded image.

        Returns:
            The decoded JSON value, or None if the response has no valid JSON.

        Raises:
            RuntimeError: If all connection attempts fail.
        """
        encoded_images: List[str] = [
            encode_image(image, max_image_size) for image in images
        ]
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "images": encoded_images,
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {"temperature": 0, "num_predict": 256},
            }
        ).encode("utf-8")
        last_error: Optional[Exception] = None
        for attempt, delay in enumerate((0.0, *self.retry_delays)):
            if delay:
                console.warning(
                    f"Vision request failed ({last_error}); retrying in "
                    f"{delay:.0f}s (attempt {attempt + 1})."
                )
                time.sleep(delay)
            request = Request(
                f"{self.url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                return extract_json(result.get("response", ""))
            except (
                OSError,
                URLError,
                TimeoutError,
                HTTPException,
                json.JSONDecodeError,
            ) as error:
                last_error = error
        raise RuntimeError(
            f"Local vision request to {self.url} failed after "
            f"{len(self.retry_delays) + 1} attempts: {last_error}"
        ) from last_error
