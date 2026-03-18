"""Image captioner: crop PDF regions and generate captions via GPT-4o vision."""
from __future__ import annotations

import asyncio
import base64
import io
import logging
from pathlib import Path

from openai import AsyncOpenAI

from doc_parser.chunker import Chunk
from doc_parser.utils.pdf_utils import pdf_page_to_image

logger = logging.getLogger(__name__)

_CAPTION_SYSTEM_PROMPT = (
    "You are a scientific figure captioning assistant. "
    "Describe the figure in 1-2 sentences for use in a document retrieval system."
)

# Minimum crop size in pixels; smaller regions are likely detection noise
_MIN_CROP_SIZE_PX: int = 50


async def _caption_single(
    chunk: Chunk,
    pdf_path: Path,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> None:
    """Crop the image region from the PDF and fill in caption + image_base64.

    Modifies `chunk` in-place. On any error, sets chunk.text = "[figure]" and logs
    a warning — never raises so that a single failure does not abort the pipeline.

    Args:
        chunk: An image-modality Chunk with a valid bbox.
        pdf_path: Path to the source PDF file.
        client: Authenticated AsyncOpenAI client.
        semaphore: Concurrency limiter for API calls.
    """
    async with semaphore:
        try:
            # Render the page at 150 DPI (fast, sufficient quality for captions)
            page_img = pdf_page_to_image(pdf_path, chunk.page - 1, dpi=150)
            w, h = page_img.size

            bbox = chunk.bbox  # [x1, y1, x2, y2] in normalised 0–1000 coords
            x1 = int(bbox[0] * w / 1000)
            y1 = int(bbox[1] * h / 1000)
            x2 = int(bbox[2] * w / 1000)
            y2 = int(bbox[3] * h / 1000)

            crop = page_img.crop((x1, y1, x2, y2))
            crop_w, crop_h = crop.size

            if crop_w < _MIN_CROP_SIZE_PX or crop_h < _MIN_CROP_SIZE_PX:
                logger.debug(
                    "Skipping tiny crop (%dx%d) for chunk %s", crop_w, crop_h, chunk.chunk_id
                )
                chunk.text = "[figure]"
                return

            # Encode crop as base64 PNG
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            # Call GPT-4o vision
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": _CAPTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            }
                        ],
                    },
                ],
                max_tokens=256,
                temperature=0.0,
            )

            caption = (response.choices[0].message.content or "").strip()
            chunk.text = caption
            chunk.caption = caption
            chunk.image_base64 = b64

            logger.debug("Captioned chunk %s: %s", chunk.chunk_id, caption[:80])

        except Exception:
            logger.warning("Caption failed for chunk %s", chunk.chunk_id, exc_info=True)
            chunk.text = "[figure]"


async def enrich_image_chunks(
    chunks: list[Chunk],
    pdf_path: Path,
    client: AsyncOpenAI,
    max_concurrent: int = 5,
) -> list[Chunk]:
    """Crop image regions from the PDF and fill in caption + image_base64 for image chunks.

    Non-image chunks are returned unchanged. Image chunks without a bbox are set to
    text="[figure]" and returned without captioning.

    Args:
        chunks: All chunks from the document (mixed modalities).
        pdf_path: Path to the source PDF file.
        client: Authenticated AsyncOpenAI client.
        max_concurrent: Maximum number of concurrent GPT-4o vision API calls.

    Returns:
        The same list (mutated in-place) with image chunks now having .text=caption
        and .image_base64=png_bytes_b64.
    """
    image_chunks = [
        c for c in chunks if c.modality == "image" and c.bbox is not None
    ]
    no_bbox = [c for c in chunks if c.modality == "image" and c.bbox is None]

    for c in no_bbox:
        logger.debug("Image chunk %s has no bbox; setting text='[figure]'", c.chunk_id)
        c.text = "[figure]"

    if not image_chunks:
        return chunks

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        _caption_single(chunk, pdf_path, client, semaphore) for chunk in image_chunks
    ]
    await asyncio.gather(*tasks)

    captioned = sum(1 for c in image_chunks if c.caption is not None)
    logger.info(
        "Captioned %d/%d image chunks from %s", captioned, len(image_chunks), pdf_path.name
    )
    return chunks
