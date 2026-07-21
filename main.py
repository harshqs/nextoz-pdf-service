from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import fitz  # PyMuPDF
import io
import subprocess
import os
import tempfile
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="NextOz PDF Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Thread pool for CPU-bound OCR work
executor = ThreadPoolExecutor(max_workers=4)


def is_tesseract_available() -> bool:
    try:
        r = subprocess.run(["tesseract", "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def ocr_page(page: fitz.Page) -> str:
    """Render a single PDF page as image and OCR it."""
    # 200 DPI is fast enough and accurate
    mat = fitz.Matrix(200 / 72, 200 / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_bytes = pix.tobytes("png")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        tmp_img.write(img_bytes)
        tmp_path = tmp_img.name

    out_path = tmp_path.replace(".png", "_ocr")
    try:
        subprocess.run(
            ["tesseract", tmp_path, out_path, "-l", "eng", "--psm", "6"],
            capture_output=True,
            timeout=30,
        )
        out_txt = out_path + ".txt"
        if os.path.exists(out_txt):
            with open(out_txt, "r", encoding="utf-8", errors="replace") as f:
                result = f.read().strip()
            os.remove(out_txt)
            return result
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return ""


def extract_page(page: fitz.Page, use_ocr: bool) -> tuple[str, bool]:
    """
    Extract text from a single page.
    Returns (text, was_ocr_used).
    Automatically decides: if normal extraction gives enough text, skip OCR.
    """
    # Fast path: native text extraction
    text = page.get_text("text").strip()
    if len(text) > 30:
        return text, False

    # Try blocks for better layout
    blocks = page.get_text("blocks")
    block_text = "\n".join(b[4] for b in blocks if isinstance(b[4], str)).strip()
    if len(block_text) > 30:
        return block_text, False

    # Page has no readable text — OCR if available
    if use_ocr:
        ocr_text = ocr_page(page)
        return ocr_text, True

    return text or block_text, False


@app.get("/health")
def health():
    return {"status": "ok", "tesseract": is_tesseract_available()}


@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """Standard extract endpoint — returns JSON when done."""
    filename = file.filename or "document"
    name_lower = filename.lower()
    is_pdf = name_lower.endswith(".pdf") or file.content_type == "application/pdf"
    is_txt = name_lower.endswith(".txt") or file.content_type == "text/plain"

    if not is_pdf and not is_txt:
        raise HTTPException(status_code=415, detail="Only PDF and TXT files are supported.")

    contents = await file.read()
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max size is 25 MB.")

    try:
        if is_txt:
            return {
                "text": contents.decode("utf-8", errors="replace").strip(),
                "name": filename,
                "pages": 1,
                "char_count": len(contents),
            }

        doc = fitz.open(stream=io.BytesIO(contents), filetype="pdf")
        num_pages = doc.page_count
        tess = is_tesseract_available()

        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, extract_page, doc[i], tess)
            for i in range(num_pages)
        ]
        results = await asyncio.gather(*tasks)
        doc.close()

        pages_text = [r[0] for r in results if r[0]]
        text = "\n\n".join(pages_text).strip()

        if not text:
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted. Try a text-based PDF or ensure Tesseract is installed.",
            )

        return {"text": text, "name": filename, "pages": num_pages, "char_count": len(text)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {str(e)}")


@app.post("/extract-stream")
async def extract_text_stream(file: UploadFile = File(...)):
    """
    Streaming extract — sends Server-Sent Events with progress.
    Each event: {"page": N, "total": T, "text": "...", "done": false}
    Final event: {"done": true, "text": "full text", "pages": T}
    """
    filename = file.filename or "document"
    name_lower = filename.lower()
    is_pdf = name_lower.endswith(".pdf") or file.content_type == "application/pdf"
    is_txt = name_lower.endswith(".txt") or file.content_type == "text/plain"

    if not is_pdf and not is_txt:
        raise HTTPException(status_code=415, detail="Only PDF and TXT files are supported.")

    contents = await file.read()
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 25 MB.")

    async def generate():
        try:
            if is_txt:
                text = contents.decode("utf-8", errors="replace").strip()
                yield f"data: {json.dumps({'done': True, 'text': text, 'pages': 1, 'name': filename})}\n\n"
                return

            doc = fitz.open(stream=io.BytesIO(contents), filetype="pdf")
            num_pages = doc.page_count
            tess = is_tesseract_available()
            all_pages: list[str] = [""] * num_pages
            loop = asyncio.get_event_loop()

            # Send total page count immediately
            yield f"data: {json.dumps({'page': 0, 'total': num_pages, 'done': False, 'status': 'starting'})}\n\n"

            # Process pages concurrently, yield progress as each finishes
            pending = {
                asyncio.ensure_future(
                    loop.run_in_executor(executor, extract_page, doc[i], tess)
                ): i
                for i in range(num_pages)
            }

            completed = 0
            while pending:
                done_set, _ = await asyncio.wait(
                    list(pending.keys()), return_when=asyncio.FIRST_COMPLETED
                )
                for fut in done_set:
                    page_idx = pending.pop(fut)
                    page_text, used_ocr = fut.result()
                    all_pages[page_idx] = page_text
                    completed += 1
                    yield f"data: {json.dumps({'page': completed, 'total': num_pages, 'done': False, 'ocr': used_ocr})}\n\n"

            doc.close()

            full_text = "\n\n".join(p for p in all_pages if p).strip()
            if not full_text:
                yield f"data: {json.dumps({'done': True, 'error': 'No text could be extracted.'})}\n\n"
            else:
                yield f"data: {json.dumps({'done': True, 'text': full_text, 'pages': num_pages, 'name': filename, 'char_count': len(full_text)})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'done': True, 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
