"""Free document conversion used by the public Tools pages and by PDF Jobs.

Nothing here touches a database or keeps a file. Every conversion works on
bytes in a temporary folder that is removed before the function returns.

Word to PDF needs LibreOffice (the `soffice` program). The Docker image tries
to install it; if it is missing, the functions raise ConversionError with a
plain message instead of crashing the app.
"""
import io
import os
import re
import shutil
import subprocess
import tempfile
import threading
import zipfile

OFFICE_EXTENSIONS = {"docx", "doc", "rtf", "odt", "txt"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif"}
MAX_PDF_PAGES_FOR_IMAGES = 40
MAX_PDF_PAGES_FOR_WORD = 60

# LibreOffice is heavy. Two at once is plenty for a free tool.
_OFFICE_SLOTS = threading.BoundedSemaphore(2)


class ConversionError(Exception):
    """A problem worth showing to the person who uploaded the file."""


def file_extension(name):
    name = os.path.basename(name or "")
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def safe_stem(name, default="document"):
    stem = os.path.splitext(os.path.basename(name or ""))[0]
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "", stem).strip(" .")[:80]
    return stem or default


def soffice_path():
    return shutil.which("soffice") or shutil.which("libreoffice")


def office_to_pdf_bytes(raw, filename="document.docx", timeout=120):
    """Word (or similar) to PDF using LibreOffice in a throwaway profile."""
    binary = soffice_path()
    if not binary:
        raise ConversionError("Word conversion is not available on this server right now. Please try again later.")
    ext = file_extension(filename)
    if ext not in OFFICE_EXTENSIONS:
        raise ConversionError("That file type is not a Word document.")
    work = tempfile.mkdtemp(prefix="tm-office-")
    try:
        source = os.path.join(work, f"input.{ext}")
        with open(source, "wb") as handle:
            handle.write(raw)
        profile = os.path.join(work, "profile")
        command = [
            binary, "--headless", "--norestore", "--nolockcheck", "--nodefault", "--nofirststartwizard",
            f"-env:UserInstallation=file://{profile}", "--convert-to", "pdf:writer_pdf_Export", "--outdir", work, source,
        ]
        env = dict(os.environ, HOME=work)
        with _OFFICE_SLOTS:
            try:
                subprocess.run(command, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
            except subprocess.TimeoutExpired as exc:
                raise ConversionError("The document took too long to convert. Try a smaller file.") from exc
        produced = os.path.join(work, "input.pdf")
        if not os.path.exists(produced):
            raise ConversionError("That document could not be converted. It may be damaged or password protected.")
        with open(produced, "rb") as handle:
            return handle.read()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _render_pdf_pages(raw, image_format, max_pages, scale=2.0):
    import pypdfium2 as pdfium

    try:
        document = pdfium.PdfDocument(raw)
    except Exception as exc:
        raise ConversionError("That PDF could not be opened. It may be damaged or password protected.") from exc
    try:
        count = len(document)
        if count < 1:
            raise ConversionError("That PDF has no pages.")
        if count > max_pages:
            raise ConversionError(f"That PDF has {count} pages. The free tool handles up to {max_pages} pages at a time.")
        pages = []
        for index in range(count):
            page = document[index]
            width, height = page.get_size()
            fit = min(scale, (16_000_000 / max(float(width * height), 1.0)) ** 0.5)
            image = page.render(scale=max(0.5, fit)).to_pil().convert("RGB")
            out = io.BytesIO()
            if image_format == "png":
                image.save(out, format="PNG", optimize=True)
            else:
                image.save(out, format="JPEG", quality=92, optimize=True)
            pages.append(out.getvalue())
        return pages
    finally:
        try:
            document.close()
        except Exception:
            pass


def pdf_to_image_list(raw, image_format="jpg", max_pages=MAX_PDF_PAGES_FOR_IMAGES):
    return _render_pdf_pages(raw, "png" if image_format == "png" else "jpg", max_pages)


def pdf_to_docx_bytes(raw):
    """PDF to an editable Word file. Scanned PDFs have no text to carry over."""
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(raw))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                raise ConversionError("That PDF is password protected.")
        pages = len(reader.pages)
        if pages > MAX_PDF_PAGES_FOR_WORD:
            raise ConversionError(f"That PDF has {pages} pages. The free tool handles up to {MAX_PDF_PAGES_FOR_WORD} pages at a time.")
        has_text = any((page.extract_text() or "").strip() for page in reader.pages[:10])
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError("That PDF could not be read. It may be damaged.") from exc
    if not has_text:
        raise ConversionError("This PDF looks like a scan, so there is no text to put into Word. Convert it to images instead, or transcribe the pages with TypeMyworDz.")
    work = tempfile.mkdtemp(prefix="tm-pdf-")
    try:
        source = os.path.join(work, "input.pdf")
        target = os.path.join(work, "output.docx")
        with open(source, "wb") as handle:
            handle.write(raw)
        try:
            from pdf2docx import Converter

            converter = Converter(source)
            try:
                converter.convert(target)
            finally:
                converter.close()
        except ImportError:
            return _pdf_text_to_docx(reader)
        except Exception:
            return _pdf_text_to_docx(reader)
        if not os.path.exists(target):
            return _pdf_text_to_docx(reader)
        with open(target, "rb") as handle:
            return handle.read()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _pdf_text_to_docx(reader):
    """Plain fallback: the text of every page, one page after another."""
    from docx import Document

    document = Document()
    for number, page in enumerate(reader.pages):
        if number:
            document.add_page_break()
        text = page.extract_text() or ""
        for block in re.split(r"\n\s*\n", text):
            block = " ".join(block.split())
            if block:
                document.add_paragraph(block)
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()


def images_to_pdf_bytes(image_blobs):
    from PIL import Image, ImageOps

    frames = []
    for blob in image_blobs:
        try:
            image = Image.open(io.BytesIO(blob))
            image.load()
        except Exception as exc:
            raise ConversionError("One of the pictures could not be read.") from exc
        frames.append(ImageOps.exif_transpose(image).convert("RGB"))
    if not frames:
        raise ConversionError("Choose at least one picture.")
    out = io.BytesIO()
    frames[0].save(out, format="PDF", save_all=True, append_images=frames[1:], resolution=150.0)
    return out.getvalue()


def image_convert_bytes(raw, image_format):
    from PIL import Image, ImageOps

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise ConversionError("That picture could not be read.") from exc
    image = ImageOps.exif_transpose(image)
    out = io.BytesIO()
    if image_format == "png":
        image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB").save(out, format="PNG", optimize=True)
    else:
        image.convert("RGB").save(out, format="JPEG", quality=92, optimize=True)
    return out.getvalue()


def _zip_named(files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files:
            archive.writestr(name, data)
    return out.getvalue()


TARGETS = {"pdf", "docx", "png", "jpg"}


def convert(filename, raw, target):
    """Return (download_name, content_type, bytes) or raise ConversionError."""
    target = (target or "").lower().replace("jpeg", "jpg")
    if target not in TARGETS:
        raise ConversionError("Choose what to convert the file into.")
    ext = file_extension(filename)
    stem = safe_stem(filename)
    is_pdf = ext == "pdf" or raw[:5] == b"%PDF-"
    if is_pdf:
        if target == "docx":
            return f"{stem}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", pdf_to_docx_bytes(raw)
        if target in ("png", "jpg"):
            pages = pdf_to_image_list(raw, target)
            mime = "image/png" if target == "png" else "image/jpeg"
            if len(pages) == 1:
                return f"{stem}.{target}", mime, pages[0]
            return f"{stem}-{target}-pages.zip", "application/zip", _zip_named([(f"{stem}-page-{i + 1:02d}.{target}", data) for i, data in enumerate(pages)])
        raise ConversionError("That file is already a PDF.")
    if ext in OFFICE_EXTENSIONS:
        pdf = office_to_pdf_bytes(raw, filename)
        if target == "pdf":
            return f"{stem}.pdf", "application/pdf", pdf
        if target in ("png", "jpg"):
            pages = pdf_to_image_list(pdf, target)
            mime = "image/png" if target == "png" else "image/jpeg"
            if len(pages) == 1:
                return f"{stem}.{target}", mime, pages[0]
            return f"{stem}-{target}-pages.zip", "application/zip", _zip_named([(f"{stem}-page-{i + 1:02d}.{target}", data) for i, data in enumerate(pages)])
        raise ConversionError("That file is already a Word document.")
    if ext in IMAGE_EXTENSIONS:
        if target == "pdf":
            return f"{stem}.pdf", "application/pdf", images_to_pdf_bytes([raw])
        if target in ("png", "jpg"):
            return f"{stem}.{target}", "image/png" if target == "png" else "image/jpeg", image_convert_bytes(raw, target)
        raise ConversionError("A picture can be turned into a PDF, PNG or JPG.")
    raise ConversionError("That file type is not supported. Use a Word document, a PDF or a picture.")
