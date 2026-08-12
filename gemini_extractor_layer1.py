"""Resilient Gemini extraction for ROJ Guard.

Primary path: current Google GenAI SDK + a supported stable Gemini model.
Fallback path: deterministic local parsing keeps the demo functional when the
external API is unavailable, rate-limited, or misconfigured.
"""

import os
import json
import re
from pathlib import Path

try:
    from google import genai
except Exception:
    genai = None

from schemas_layer1 import GeminiExtractionResult
from time_utils import current_date

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
ENV_MODEL = os.environ.get("GEMINI_MODEL", "").strip()

# Gemini 2.0 Flash is shut down as of 2026. Do not let an old .env value keep
# breaking the app. Prefer the current stable Flash model and retain a short
# fallback chain for future endpoint changes.
DECOMMISSIONED_MODELS = {"gemini-2.0-flash", "gemini-2.0-flash-lite"}
MODEL_CANDIDATES = []
if ENV_MODEL and ENV_MODEL not in DECOMMISSIONED_MODELS:
    MODEL_CANDIDATES.append(ENV_MODEL)
for _m in (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
):
    if _m not in MODEL_CANDIDATES:
        MODEL_CANDIDATES.append(_m)


def _client():
    if genai is None or not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


FIELD_SCHEMAS = {
    "PO": {
        "po_number": "string",
        "vendor_name": "string",
        "material_description": "string",
        "sku": "string or null",
        "quantity": "number",
        "order_date": "YYYY-MM-DD",
        "promised_ship_date": "YYYY-MM-DD",
        "unit_price": "number or null",
        "incoterms": "string or null",
    },
    "SUBMITTAL": {
        "material_description": "string",
        "spec_section": "string",
        "approval_status": "one of: pending, approved, rejected",
        "submitted_date": "YYYY-MM-DD",
        "approved_date": "YYYY-MM-DD or null",
    },
    "VENDOR_EMAIL": {
        "vendor_name": "string",
        "material_description": "string or null",
        "comm_type": "one of: status_update, delay_notice, confirmation",
        "message_date": "YYYY-MM-DD",
        "extracted_summary": "string, 1-2 sentence summary",
        "delay_days_mentioned": "integer or null",
    },
    "SHIPPING": {
        "material_description": "string or null",
        "vendor_name": "string or null",
        "carrier": "string",
        "tracking_number": "string or null",
        "current_location": "string",
        "shipped_date": "YYYY-MM-DD or null",
        "estimated_arrival": "YYYY-MM-DD or null",
        "actual_delivered_date": "YYYY-MM-DD or null",
        "status": "one of: in_transit, delayed, delivered",
    },
    "SCHEDULE": {
        "task_name": "string",
        "material_description": "string or null",
        "roj_date": "YYYY-MM-DD",
        "float_days": "integer or null",
        "is_critical_path": "boolean",
    },
}


def _build_prompt(doc_type: str) -> str:
    schema = FIELD_SCHEMAS.get(doc_type)
    if not schema:
        raise ValueError(f"Unknown doc_type: {doc_type}")
    return f"""
You are a document-extraction agent for a construction supply-chain system.
You will be given a document of type: {doc_type}.

Extract only facts present in the document. Do not infer or invent values.
Return JSON with this exact top-level structure:
{{
  "doc_type": "{doc_type}",
  "confidence": <float between 0 and 1>,
  "extracted_fields": {{ ... }},
  "notes": <string or null>
}}

Required extracted_fields schema:
{json.dumps(schema, indent=2)}

Rules:
- Missing fields must be null.
- Dates must use YYYY-MM-DD.
- confidence should reflect extraction certainty.
- Return valid JSON only.
"""


def _clean_sentence(text: str, max_len: int = 360) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "…"


def _extract_date(text: str):
    m = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text or "")
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    m = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](20\d{2})\b", text or "")
    if m:
        d, mo, y = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    return str(current_date())


def _local_vendor_email(text: str) -> GeminiExtractionResult:
    raw = _clean_sentence(text, 2000)
    lower = raw.lower()

    delay = None
    for pattern in [
        r"(?:delay(?:ed)?|slip(?:ped)?|late|postpon(?:ed|ement)|behind)[^\d]{0,45}(\d{1,3})\s*(?:calendar\s*)?days?",
        r"(\d{1,3})\s*(?:calendar\s*)?days?[^.]{0,45}(?:delay|late|slip)",
    ]:
        m = re.search(pattern, raw, flags=re.I)
        if m:
            delay = int(m.group(1))
            break

    vendor = None
    for pattern in [
        r"^\s*([A-Z][A-Za-z0-9&.,'()\- ]{2,70}?)\s+(?:has|have)\s+(?:informed|confirmed|advised|reported|notified|stated)",
        r"(?:from|vendor|supplier)\s*[:\-]\s*([A-Z][A-Za-z0-9&.,'()\- ]{2,70})(?:\n|\.|,)",
    ]:
        m = re.search(pattern, raw, flags=re.I)
        if m:
            vendor = m.group(1).strip(" ,.-")
            break

    material = None
    for pattern in [
        r"production\s+of\s+(.{3,120}?)\s+(?:has|have|is|are)\s+(?:been\s+)?(?:delayed|postponed|affected)",
        r"delivery\s+of\s+(.{3,120}?)\s+(?:has|have|is|are)",
        r"(?:material|item|product)\s*[:\-]\s*(.{3,120}?)(?:\.|\n|;)",
        r"for\s+(.{3,120}?)\s+(?:is|has|will|was)\s+(?:delayed|late|slipping|postponed)",
    ]:
        m = re.search(pattern, raw, flags=re.I)
        if m:
            material = m.group(1).strip(" ,.-")
            break

    is_delay = bool(delay) or any(k in lower for k in ("delay", "late", "slip", "postpon", "constraint"))
    is_confirmation = any(k in lower for k in ("confirm", "on schedule", "on track")) and not is_delay
    comm_type = "delay_notice" if is_delay else ("confirmation" if is_confirmation else "status_update")

    found = sum(x is not None for x in (vendor, material, delay))
    confidence = {0: 0.45, 1: 0.62, 2: 0.78, 3: 0.90}[found]
    return GeminiExtractionResult(
        doc_type="VENDOR_EMAIL",
        confidence=confidence,
        extracted_fields={
            "vendor_name": vendor,
            "material_description": material,
            "comm_type": comm_type,
            "message_date": _extract_date(raw),
            "extracted_summary": _clean_sentence(raw),
            "delay_days_mentioned": delay,
        },
        notes="Fallback parser used. Review extracted fields before applying.",
    )


def _local_generic(text: str, doc_type: str) -> GeminiExtractionResult:
    if doc_type == "VENDOR_EMAIL":
        return _local_vendor_email(text)
    return GeminiExtractionResult(
        doc_type=doc_type,
        confidence=0.0,
        extracted_fields={},
        notes="AI provider unavailable. Review this document manually.",
    )


def _parse_response(raw_response_text: str, doc_type: str, model_used: str = None) -> GeminiExtractionResult:
    try:
        cleaned = (raw_response_text or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        result = GeminiExtractionResult(**data)
        if result.doc_type == "UNKNOWN":
            result.doc_type = doc_type
        if model_used:
            result.notes = f"Extracted with {model_used}."
        return result
    except Exception:
        return GeminiExtractionResult(
            doc_type=doc_type,
            confidence=0.0,
            extracted_fields={},
            notes="AI response could not be parsed; review manually.",
        )


def _generate(contents, config=None):
    client = _client()
    if client is None:
        raise RuntimeError("Gemini client is not configured")

    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config or {},
            )
            return response, model_name
        except Exception as exc:
            last_error = exc
            msg = str(exc).lower()
            # Try the next supported model only when this endpoint/model is gone.
            if "404" in msg or "not_found" in msg or "not found" in msg or "unsupported" in msg:
                continue
            raise
    raise last_error or RuntimeError("No supported Gemini model was available")


def _gemini_text(text: str, doc_type: str) -> GeminiExtractionResult:
    prompt = _build_prompt(doc_type) + f"\n\nDOCUMENT TEXT:\n{text}"
    response, model_used = _generate(
        prompt,
        config={"response_mime_type": "application/json"},
    )
    return _parse_response(response.text, doc_type, model_used)


def extract_from_text(raw_text: str, doc_type: str) -> GeminiExtractionResult:
    doc_type = doc_type.upper()
    text = raw_text or ""

    if genai is None or not GEMINI_API_KEY:
        return _local_generic(text, doc_type)

    try:
        result = _gemini_text(text, doc_type)
        if doc_type == "VENDOR_EMAIL" and (not result.extracted_fields or result.confidence <= 0.0):
            return _local_vendor_email(text)
        return result
    except Exception:
        return _local_generic(text, doc_type)


def extract_from_pdf(file_path: str, doc_type: str) -> GeminiExtractionResult:
    doc_type = doc_type.upper()

    # Prefer local text extraction first: it is fast, deterministic and avoids
    # uploading ordinary text PDFs unnecessarily.
    text = ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        text = ""

    if text.strip():
        return extract_from_text(text, doc_type)

    # If the PDF is image-based / has no extractable text, use Gemini file input.
    if genai is not None and GEMINI_API_KEY:
        try:
            client = _client()
            uploaded_file = client.files.upload(file=file_path)
            prompt = _build_prompt(doc_type)
            response, model_used = _generate(
                [prompt, uploaded_file],
                config={"response_mime_type": "application/json"},
            )
            result = _parse_response(response.text, doc_type, model_used)
            if result.extracted_fields and result.confidence > 0:
                return result
        except Exception:
            pass

    return GeminiExtractionResult(
        doc_type=doc_type,
        confidence=0.0,
        extracted_fields={},
        notes="PDF received, but AI extraction was unavailable and no extractable text was found. Route to human review.",
    )
