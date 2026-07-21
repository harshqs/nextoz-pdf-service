from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import io

app = FastAPI(title="NextOz PDF Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production to your Next.js domain
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    filename = file.filename or ""
    is_pdf = filename.lower().endswith(".pdf") or file.content_type == "application/pdf"
    is_txt = filename.lower().endswith(".txt") or file.content_type == "text/plain"

    if not is_pdf and not is_txt:
        raise HTTPException(status_code=415, detail="Only PDF and TXT files are supported.")

    contents = await file.read()

    if len(contents) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=413, detail="File too large. Max size is 20 MB.")

    try:
        if is_txt:
            text = contents.decode("utf-8", errors="replace")
        else:
            # PyMuPDF (fitz) — handles encrypted, compressed, and complex PDFs
            doc = fitz.open(stream=io.BytesIO(contents), filetype="pdf")
            pages = []
            for page in doc:
                pages.append(page.get_text("text"))
            doc.close()
            text = "\n".join(pages).strip()

        if not text.strip():
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted. This may be a scanned/image-based PDF.",
            )

        return {"text": text.strip(), "name": filename, "pages": len(pages) if is_pdf else 1}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {str(e)}")
