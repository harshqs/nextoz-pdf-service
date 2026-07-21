from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import io
import subprocess
import sys

app = FastAPI(title="NextOz PDF Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def extract_with_pymupdf(contents: bytes) -> tuple[str, int]:
    """Try all PyMuPDF extraction strategies."""
    doc = fitz.open(stream=io.BytesIO(contents), filetype="pdf")
    num_pages = doc.page_count
    pages_text = []

    for page in doc:
        # Strategy 1: standard text
        text = page.get_text("text").strip()

        # Strategy 2: blocks (preserves layout better)
        if not text:
            blocks = page.get_text("blocks")
            text = "\n".join(b[4] for b in blocks if isinstance(b[4], str)).strip()

        # Strategy 3: dict (gets every text span)
        if not text:
            data = page.get_text("dict")
            spans = []
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        spans.append(span.get("text", ""))
            text = " ".join(spans).strip()

        if text:
            pages_text.append(text)

    doc.close()
    return "\n\n".join(pages_text).strip(), num_pages


def ocr_with_tesseract(contents: bytes) -> str:
    """Render PDF pages as images and OCR them using tesseract CLI."""
    doc = fitz.open(stream=io.BytesIO(contents), filetype="pdf")
    results = []

    for page_num, page in enumerate(doc):
        # Render at 300 DPI for good OCR quality
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_bytes = pix.tobytes("png")

        # Write to temp file and call tesseract
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            tmp_img.write(img_bytes)
            tmp_img_path = tmp_img.name

        out_path = tmp_img_path.replace(".png", "_out")
        try:
            result = subprocess.run(
                ["tesseract", tmp_img_path, out_path, "-l", "eng", "--psm", "6"],
                capture_output=True, timeout=60
            )
            out_txt = out_path + ".txt"
            if os.path.exists(out_txt):
                with open(out_txt, "r", encoding="utf-8", errors="replace") as f:
                    results.append(f.read().strip())
                os.remove(out_txt)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        finally:
            if os.path.exists(tmp_img_path):
                os.remove(tmp_img_path)

    doc.close()
    return "\n\n".join(results).strip()


def is_tesseract_available() -> bool:
    try:
        r = subprocess.run(["tesseract", "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tesseract": is_tesseract_available(),
    }


@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    filename = file.filename or "document"
    name_lower = filename.lower()
    is_pdf = name_lower.endswith(".pdf") or file.content_type == "application/pdf"
    is_txt = name_lower.endswith(".txt") or file.content_type == "text/plain"

    if not is_pdf and not is_txt:
        raise HTTPException(
            status_code=415,
            detail="Only PDF and TXT files are supported."
        )

    contents = await file.read()

    if len(contents) > 25 * 1024 * 1024:  # 25 MB limit
        raise HTTPException(status_code=413, detail="File too large. Max size is 25 MB.")

    try:
        if is_txt:
            text = contents.decode("utf-8", errors="replace").strip()
            num_pages = 1
        else:
            # First try normal text extraction
            text, num_pages = extract_with_pymupdf(contents)

            # If still empty → try OCR fallback
            if not text and is_tesseract_available():
                text = ocr_with_tesseract(contents)

        if not text:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No text could be extracted from this PDF. "
                    "It appears to be a scanned/image-based PDF. "
                    "Please install Tesseract OCR on the server for image PDF support, "
                    "or try a text-based PDF."
                ),
            )

        return {
            "text": text,
            "name": filename,
            "pages": num_pages,
            "char_count": len(text),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {str(e)}")
