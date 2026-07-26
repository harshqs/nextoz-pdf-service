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
import time
from concurrent.futures import ThreadPoolExecutor

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace import Status, StatusCode

# 1. Configure OpenTelemetry Resources
resource = Resource.create({
    ResourceAttributes.SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "nextoz-pdf-service"),
    ResourceAttributes.SERVICE_VERSION: os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
})

# 2. Configure Tracing
otlp_traces_endpoint = os.getenv(
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
)
tracer_provider = TracerProvider(resource=resource)
trace_exporter = OTLPSpanExporter(endpoint=otlp_traces_endpoint)
tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer("nextoz-pdf-service")

# 3. Configure Metrics
otlp_metrics_endpoint = os.getenv(
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/metrics")
)
metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint=otlp_metrics_endpoint)
)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter("nextoz-pdf-service")

# Custom Metrics
pages_processed_counter = meter.create_counter(
    "nextoz_pdf_pages_processed_total",
    description="Total number of PDF/TXT document pages processed",
    unit="1"
)
ocr_triggered_counter = meter.create_counter(
    "nextoz_pdf_ocr_triggered_total",
    description="Total number of times Tesseract OCR was invoked",
    unit="1"
)
extraction_duration_histogram = meter.create_histogram(
    "nextoz_pdf_extraction_duration_seconds",
    description="Duration of document extraction in seconds",
    unit="s"
)

app = FastAPI(title="NextOz PDF Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://nextoz.vercel.app",
        "https://*.vercel.app",
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Auto-instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

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
    with tracer.start_as_current_span("pdf.ocr_page") as span:
        ocr_triggered_counter.add(1)
        span.set_attribute("nextoz.ocr_engine", "tesseract")
        
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
                span.set_attribute("nextoz.ocr_output_length", len(result))
                return result
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return ""


def extract_page(page: fitz.Page, use_ocr: bool) -> tuple[str, bool]:
    """
    Extract text from a single page.
    Returns (text, was_ocr_used).
    """
    with tracer.start_as_current_span("pdf.extract_page") as span:
        span.set_attribute("pdf.page_number", page.number)
        
        # Fast path: native text extraction
        text = page.get_text("text").strip()
        if len(text) > 30:
            span.set_attribute("pdf.extraction_method", "native_text")
            span.set_attribute("pdf.extracted_char_count", len(text))
            return text, False

        # Try blocks for better layout
        blocks = page.get_text("blocks")
        block_text = "\n".join(b[4] for b in blocks if isinstance(b[4], str)).strip()
        if len(block_text) > 30:
            span.set_attribute("pdf.extraction_method", "native_blocks")
            span.set_attribute("pdf.extracted_char_count", len(block_text))
            return block_text, False

        # Page has no readable text — OCR if available
        if use_ocr:
            span.set_attribute("pdf.extraction_method", "ocr")
            ocr_text = ocr_page(page)
            span.set_attribute("pdf.extracted_char_count", len(ocr_text))
            return ocr_text, True

        span.set_attribute("pdf.extraction_method", "fallback")
        res = text or block_text
        span.set_attribute("pdf.extracted_char_count", len(res))
        return res, False


@app.get("/health")
def health():
    return {"status": "ok", "tesseract": is_tesseract_available()}


@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """Standard extract endpoint — returns JSON when done."""
    start_time = time.time()
    with tracer.start_as_current_span("pdf.extract_full") as span:
        filename = file.filename or "document"
        name_lower = filename.lower()
        is_pdf = name_lower.endswith(".pdf") or file.content_type == "application/pdf"
        is_txt = name_lower.endswith(".txt") or file.content_type == "text/plain"

        span.set_attribute("file.name", filename)
        span.set_attribute("file.is_pdf", is_pdf)
        span.set_attribute("file.is_txt", is_txt)

        if not is_pdf and not is_txt:
            span.set_status(Status(StatusCode.ERROR, "Unsupported file type"))
            raise HTTPException(status_code=415, detail="Only PDF and TXT files are supported.")

        contents = await file.read()
        span.set_attribute("file.size_bytes", len(contents))

        if len(contents) > 25 * 1024 * 1024:
            span.set_status(Status(StatusCode.ERROR, "File too large"))
            raise HTTPException(status_code=413, detail="File too large. Max size is 25 MB.")

        try:
            if is_txt:
                text = contents.decode("utf-8", errors="replace").strip()
                pages_processed_counter.add(1)
                extraction_duration_histogram.record(time.time() - start_time)
                return {
                    "text": text,
                    "name": filename,
                    "pages": 1,
                    "char_count": len(contents),
                }

            doc = fitz.open(stream=io.BytesIO(contents), filetype="pdf")
            num_pages = doc.page_count
            span.set_attribute("pdf.page_count", num_pages)
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

            pages_processed_counter.add(num_pages)
            extraction_duration_histogram.record(time.time() - start_time)

            if not text:
                span.set_status(Status(StatusCode.ERROR, "No text extracted"))
                raise HTTPException(
                    status_code=422,
                    detail="No text could be extracted. Try a text-based PDF or ensure Tesseract is installed.",
                )

            return {"text": text, "name": filename, "pages": num_pages, "char_count": len(text)}

        except HTTPException:
            raise
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise HTTPException(status_code=500, detail=f"Failed to parse file: {str(e)}")


@app.post("/extract-stream")
async def extract_text_stream(file: UploadFile = File(...)):
    """Streaming extract — sends Server-Sent Events with progress."""
    start_time = time.time()
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
        with tracer.start_as_current_span("pdf.extract_stream") as span:
            span.set_attribute("file.name", filename)
            span.set_attribute("file.size_bytes", len(contents))
            try:
                if is_txt:
                    text = contents.decode("utf-8", errors="replace").strip()
                    pages_processed_counter.add(1)
                    extraction_duration_histogram.record(time.time() - start_time)
                    yield f"data: {json.dumps({'done': True, 'text': text, 'pages': 1, 'name': filename})}\n\n"
                    return

                doc = fitz.open(stream=io.BytesIO(contents), filetype="pdf")
                num_pages = doc.page_count
                span.set_attribute("pdf.page_count", num_pages)
                tess = is_tesseract_available()
                all_pages: list[str] = [""] * num_pages
                loop = asyncio.get_event_loop()

                yield f"data: {json.dumps({'page': 0, 'total': num_pages, 'done': False, 'status': 'starting'})}\n\n"

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
                pages_processed_counter.add(num_pages)
                extraction_duration_histogram.record(time.time() - start_time)

                if not full_text:
                    span.set_status(Status(StatusCode.ERROR, "No text extracted"))
                    yield f"data: {json.dumps({'done': True, 'error': 'No text could be extracted.'})}\n\n"
                else:
                    span.set_status(Status(StatusCode.OK))
                    yield f"data: {json.dumps({'done': True, 'text': full_text, 'pages': num_pages, 'name': filename, 'char_count': len(full_text)})}\n\n"

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                yield f"data: {json.dumps({'done': True, 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
