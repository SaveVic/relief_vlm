"""
relief.client
=============
Thin wrapper around `replicate.Client`: upload images, create a prediction,
wait for it to finish, and return the raw text together with the untouched
Prediction object.

The client deliberately knows nothing about JSON, schemas, or format repair.
It returns a `RawResponse` holding the text and the full prediction object;
`relief.analyze` decides how to parse it. That separation is what makes the
three-layer storage (raw / parsed / validated) possible without duplicating
the calling logic.

Images are uploaded through the Files API rather than as data URIs. Data URIs
are only recommended for files under 1 MB, and three 1536 px images in one
request easily exceed that. Uploads are cached by content hash, so the same
image is not re-uploaded across replicates or conditions.
"""

from __future__ import annotations

import base64
import io
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import replicate
from replicate.exceptions import ReplicateError

from .models import base_name, build_model_input, get_spec
from .usage import UsageTokens, extract_usage

logger = logging.getLogger(__name__)


def to_data_uri(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def join_output(output: Any) -> str:
    """Language models on Replicate return either a list of text chunks or a string."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, (list, tuple)):
        return "".join(str(x) for x in output)
    return str(output)


@dataclass
class RawResponse:
    """Result of a single prediction, not parsed at all."""

    text: str
    prediction: Any
    usage: UsageTokens
    model_input: Dict[str, Any] = field(default_factory=dict)


class ReplicateClient:
    """One client per model, reused across panels and conditions."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: int = 300,
        max_retries: int = 2,
        upload_mode: str = "files",
        poll_interval: float = 1.0,
        extra_input: Optional[Dict[str, Any]] = None,
        api_token: Optional[str] = None,
    ):
        if upload_mode not in ("files", "data_uri"):
            raise ValueError("upload_mode must be 'files' or 'data_uri'")

        self.model = model
        self.spec = get_spec(model)
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens is not None else self.spec.max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.upload_mode = upload_mode
        self.poll_interval = poll_interval
        self.extra_input = dict(extra_input or {})
        # api_token=None -> read from the REPLICATE_API_TOKEN env variable
        self.client = replicate.Client(api_token=api_token)
        self._uploads: Dict[str, Tuple[str, Optional[str]]] = {}  # sha256 -> (url, file_id)

    # ---- images -----------------------------------------------------------

    def upload_images(
        self, files: Sequence[Tuple[str, bytes]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        [(label, jpeg_bytes)] -> (URLs/URIs ready to send, metadata for artifacts).

        The metadata holds the label, hash, size, and URL. It goes into
        `request.json`, so it can later be verified that a call really
        received the images it claims to have received.
        """
        import hashlib

        urls: List[str] = []
        meta: List[Dict[str, Any]] = []

        for label, data in files:
            digest = hashlib.sha256(data).hexdigest()
            info = {
                "label": label,
                "sha256": digest,
                "bytes": len(data),
                "mode": self.upload_mode,
            }

            if self.upload_mode == "data_uri":
                uri = to_data_uri(data)
                if len(uri) > 1_000_000:
                    logger.warning(
                        "[%s] data URI is %.2f MB, above the 1 MB Replicate "
                        "recommends. Consider upload_mode='files'.",
                        label, len(uri) / 1e6,
                    )
                urls.append(uri)
                meta.append(info)
                continue

            if digest in self._uploads:
                url, file_id = self._uploads[digest]
                logger.debug("[%s] reusing upload (cache hit)", label)
                urls.append(url)
                meta.append({**info, "url": url, "file_id": file_id, "cached": True})
                continue

            uploaded = self._with_retry(
                lambda: self.client.files.create(
                    io.BytesIO(data),
                    filename=f"{label.lower()}_{digest[:8]}.jpg",
                    content_type="image/jpeg",
                ),
                f"upload image {label}",
            )
            url = uploaded.urls["get"]
            self._uploads[digest] = (url, uploaded.id)
            logger.info("[%s] uploaded to Files API (%.1f KB, id=%s)",
                        label, len(data) / 1024, uploaded.id)
            urls.append(url)
            meta.append({**info, "url": url, "file_id": uploaded.id, "cached": False})

        return urls, meta

    def _with_retry(self, fn, what: str):
        """
        Run `fn`, retrying on transient errors.

        Used for image uploads. Without it, a single `ConnectError` on the
        first upload kills the cell, and since the first upload happens on the
        coldest connection, the first call of every model loop is the one most
        likely to hit it.

        Permanent errors (HTTP 4xx other than 429) are raised without retry:
        repeating the call would not change the outcome.
        """
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except ReplicateError as exc:
                status = getattr(exc, "status", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                last_error = exc
            except (httpx.HTTPError, TimeoutError) as exc:
                last_error = exc

            if attempt < self.max_retries:
                delay = min(2 ** (attempt + 1), 30) + random.uniform(0, 1)
                logger.warning("%s failed (%s: %s); retrying in %.1f s",
                               what, type(last_error).__name__, last_error, delay)
                time.sleep(delay)

        raise RuntimeError(
            f"{what} failed after {self.max_retries + 1} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def cleanup_uploads(self) -> None:
        """Delete the files this client uploaded from Replicate's Files API."""
        for digest, (_, file_id) in list(self._uploads.items()):
            if file_id:
                try:
                    self.client.files.delete(file_id)
                except Exception as exc:  # noqa: BLE001 - cleanup must never fail the experiment
                    logger.warning("failed to delete file %s: %s: %s",
                                   file_id, type(exc).__name__, exc)
            self._uploads.pop(digest, None)

    # ---- predictions ------------------------------------------------------

    def _run_once(self, model_input: Dict[str, Any]):
        prediction = self.client.predictions.create(
            model=base_name(self.model), input=model_input
        )
        deadline = time.monotonic() + self.timeout
        while prediction.status not in ("succeeded", "failed", "canceled"):
            if time.monotonic() > deadline:
                try:
                    prediction.cancel()
                except Exception:  # noqa: BLE001
                    pass
                raise TimeoutError(
                    f"prediction {prediction.id} exceeded the {self.timeout}s limit and was cancelled"
                )
            time.sleep(self.poll_interval)
            prediction.reload()
        return prediction

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        images: List[str],
    ) -> RawResponse:
        """
        One model call, retrying on transient errors.

        Permanent errors (HTTP 4xx other than 429: bad token, invalid input)
        are raised immediately, since repeating the call would not change the
        outcome and every attempt still costs time.
        """
        model_input = build_model_input(
            self.model, system_prompt, user_prompt, images,
            self.temperature, self.max_tokens, self.extra_input,
        )
        last_error: Optional[BaseException] = None

        for attempt in range(self.max_retries + 1):
            try:
                prediction = self._run_once(model_input)
            except ReplicateError as exc:
                status = getattr(exc, "status", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                last_error = exc
            except (httpx.HTTPError, TimeoutError) as exc:
                last_error = exc
            else:
                if prediction.status == "succeeded":
                    return RawResponse(
                        text=join_output(prediction.output),
                        prediction=prediction,
                        usage=extract_usage(prediction, logger=logger),
                        model_input=model_input,
                    )
                last_error = RuntimeError(
                    f"prediction {prediction.id} has status '{prediction.status}': "
                    f"{prediction.error}"
                )

            if attempt < self.max_retries:
                delay = min(2 ** (attempt + 1), 30) + random.uniform(0, 1)
                logger.warning(
                    "API attempt %d failed (%s: %s); retrying in %.1f s",
                    attempt + 1, type(last_error).__name__, last_error, delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"failed after {self.max_retries + 1} API attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error
