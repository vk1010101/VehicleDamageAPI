"""
damage_service.py
-----------------
Local vehicle damage assessment pipeline:
  1. YOLODamageDetector  — (optional) draws bounding boxes if a car-damage .pt model is available
  2. QwenVLAssessor      — sends image to Qwen 3.5 via Ollama, returns structured JSON report
  3. assess_damage()     — top-level function called by the Flask route

YOLO is gracefully optional. If no .pt file exists at YOLO_MODEL_PATH, the pipeline skips
detection and sends only the original image to Qwen, which is more than capable on its own.
To enable YOLO later, drop a car-damage .pt file into the project root and set YOLO_MODEL_PATH.
"""

import base64
import json
import logging
import os
import re
import time
from io import BytesIO

import requests
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — can be overridden from app.py via module attributes before first call
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_MODEL     = "gemma3:4b"        # vision model — can actually see images
YOLO_MODEL_PATH  = "car_damage.pt"    # path to a YOLO .pt weights file (optional)
VL_MAX_DIM       = 1024              # downscale before sending to Qwen VL
VL_JPEG_QUALITY  = 85                # JPEG quality for VL payload
VL_NUM_PREDICT   = 1024              # ~250 tokens per image × 4 images

# Severity order for sorting / overall roll-up
_SEVERITY_RANK = {"None": 0, "Minor": 1, "Moderate": 2, "Severe": 3}


# ---------------------------------------------------------------------------
# YOLO — optional car-damage detector
# ---------------------------------------------------------------------------
class YOLODamageDetector:
    _model = None
    _available = None   # True / False once checked
    _device = "cpu"     # always CPU — GPU reserved for Ollama

    @classmethod
    def is_available(cls) -> bool:
        """Check once whether a YOLO .pt file exists."""
        if cls._available is None:
            cls._available = os.path.isfile(YOLO_MODEL_PATH)
            if cls._available:
                logger.info("YOLO model found at %s — detection enabled.", YOLO_MODEL_PATH)
            else:
                logger.info("No YOLO model at %s — running Qwen-only mode.", YOLO_MODEL_PATH)
        return cls._available

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            from ultralytics import YOLO
            logger.info("Loading YOLO model: %s", YOLO_MODEL_PATH)
            cls._model = YOLO(YOLO_MODEL_PATH)
            # Always use CPU for YOLO — it's a tiny model (~6MB), runs in ~0.5s on CPU.
            # This avoids CUDA conflicts with Ollama which owns the GPU for Gemma.
            cls._device = "cpu"
            logger.info("YOLO running on CPU (avoids GPU contention with Ollama).")
            logger.info("YOLO model loaded.")
        return cls._model

    @classmethod
    def detect(cls, image_bytes: bytes) -> tuple[bytes, list[dict]]:
        """
        Run YOLO inference on image_bytes.

        Returns:
            annotated_bytes : JPEG bytes of the image with bounding boxes drawn
            detections      : list of {class_name, confidence, bbox: [x1,y1,x2,y2]}
        """
        import cv2
        import numpy as np

        model = cls._get_model()

        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        np_img  = np.array(pil_img)

        results = model(np_img, verbose=False, device=cls._device)
        result  = results[0]

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class_name" : model.names[cls_id],
                "confidence" : round(float(box.conf[0]), 2),
                "bbox"       : [round(float(v)) for v in box.xyxy[0].tolist()],
            })

        annotated_np  = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_np, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_rgb)

        buf = BytesIO()
        annotated_pil.save(buf, format="JPEG", quality=85)
        annotated_bytes = buf.getvalue()

        logger.info("YOLO detected %d region(s) in image.", len(detections))
        return annotated_bytes, detections


# ---------------------------------------------------------------------------
# Qwen VL — damage report via Ollama
# ---------------------------------------------------------------------------
class QwenVLAssessor:

    @staticmethod
    def _to_b64(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("ascii")

    @staticmethod
    def _downscale_for_vl(image_bytes: bytes) -> bytes:
        """
        Keep the VL request fast by resizing large images before base64 upload.
        """
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        max_dim = max(w, h)
        if max_dim <= VL_MAX_DIM:
            return image_bytes

        scale = VL_MAX_DIM / float(max_dim)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=VL_JPEG_QUALITY, optimize=True)
        return buf.getvalue()

    @staticmethod
    def _build_prompt(image_label: str, detections: list[dict] | None) -> str:
        if detections:
            det_lines = "\n".join(
                f"  - {d['class_name']} (confidence {d['confidence']:.0%}) at bbox {d['bbox']}"
                for d in detections
            )
            det_section = f"A detection model flagged these regions:\n{det_lines}"
        else:
            det_section = ""

        prompt = f"""You are a vehicle damage assessment AI for an insurance company.
You are given one photo of a vehicle side ("{image_label}").
Examine the image for any visible damage — scratches, dents, cracks, broken parts, paint damage, deformation, or missing components.

{det_section}

Respond ONLY with a single valid JSON object — no markdown, no explanation, no extra text.
Use exactly this schema:
{{
  "overall_severity": "None|Minor|Moderate|Severe",
  "regions": [
    {{"location": "...", "damage_type": "...", "severity": "Minor|Moderate|Severe"}}
  ],
  "repair_category": "...",
  "claim_recommendation": "Approve|Investigate|Reject",
  "notes": "..."
}}

If no damage is visible, return overall_severity "None" with an empty regions array."""
        return prompt

    @classmethod
    def assess(
        cls,
        original_bytes:  bytes,
        annotated_bytes: bytes | None,
        detections:      list[dict] | None,
        image_label:     str,
    ) -> dict:
        """
        Send the original image to Qwen 3.5 via Ollama and return the parsed damage report.
        YOLO detections are passed as text context (not as a second image) to keep inference fast.
        """
        prompt = cls._build_prompt(image_label, detections)

        t0 = time.perf_counter()
        vl_bytes = cls._downscale_for_vl(original_bytes)
        prep_ms = round((time.perf_counter() - t0) * 1000)

        # Send only the original image — YOLO results are injected as text in the prompt
        payload = {
            "model"  : OLLAMA_MODEL,
            "prompt" : prompt,
            "images" : [cls._to_b64(vl_bytes)],
            "stream" : False,
            "options": {
                "temperature": 0.1,
                "num_predict": VL_NUM_PREDICT,
            },
        }

        try:
            t1 = time.perf_counter()
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=600,   # generous timeout: first call loads model into VRAM
            )
            resp.raise_for_status()
            raw_text = resp.json().get("response", "")
            infer_ms = round((time.perf_counter() - t1) * 1000)
            logger.info("Qwen raw response for %s: %s", image_label, raw_text[:200])
        except Exception as exc:
            logger.exception("Ollama request failed: %s", exc)
            report = cls._fallback_report(image_label, detections)
            report["_timings_ms"] = {"vl_prep": prep_ms, "vl_infer": None}
            return report

        report = cls._parse_json(raw_text)
        if report is None:
            logger.warning("Could not parse Qwen response as JSON. Raw: %s", raw_text[:500])
            report = cls._fallback_report(image_label, detections)
            report["_timings_ms"] = {"vl_prep": prep_ms, "vl_infer": infer_ms}
            return report

        report["image_label"] = image_label
        report["_timings_ms"] = {"vl_prep": prep_ms, "vl_infer": infer_ms}
        return report

    @classmethod
    def assess_batch(cls, entries: list[dict]) -> list[dict]:
        """
        Send ALL images in ONE Ollama call. Returns a list of report dicts.
        Each entry: {label, original_bytes, detections}
        """
        # Build per-image description lines
        per_image = []
        for i, e in enumerate(entries, 1):
            det_text = ""
            if e.get("detections"):
                det_text = " Detected: " + ", ".join(
                    f"{d['class_name']} ({d['confidence']:.0%})" for d in e["detections"]
                )
            per_image.append(f"  Image {i}: \"{e['label']}\".{det_text}")

        prompt = f"""You are a vehicle damage assessment AI for an insurance company.
You are given {len(entries)} photos of a vehicle (one per side):
{chr(10).join(per_image)}

Examine ALL images for visible damage — scratches, dents, cracks, broken parts, paint damage, deformation, or missing components.

Respond ONLY with a valid JSON array — no markdown, no explanation.
One object per image, same order. Schema:
[{{"image_label":"...","overall_severity":"None|Minor|Moderate|Severe","regions":[{{"location":"...","damage_type":"...","severity":"Minor|Moderate|Severe"}}],"repair_category":"...","claim_recommendation":"Approve|Investigate|Reject","notes":"..."}}]

If no damage for an image, use overall_severity "None" with empty regions."""

        images_b64 = [
            cls._to_b64(cls._downscale_for_vl(e["original_bytes"]))
            for e in entries
        ]

        payload = {
            "model"  : OLLAMA_MODEL,
            "prompt" : prompt,
            "images" : images_b64,
            "stream" : False,
            "options": {"temperature": 0.1, "num_predict": VL_NUM_PREDICT},
        }

        try:
            t0 = time.perf_counter()
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=600,
            )
            resp.raise_for_status()
            raw_text = resp.json().get("response", "")
            logger.info("Batch Gemma response in %.1fs: %s", time.perf_counter() - t0, raw_text[:300])
        except Exception as exc:
            logger.exception("Ollama batch request failed: %s", exc)
            return [cls._fallback_report(e["label"], e.get("detections")) for e in entries]

        reports = cls._parse_json_array(raw_text)
        if reports is None or len(reports) != len(entries):
            logger.warning("Batch parse failed or count mismatch. Got %s, expected %d.",
                           len(reports) if reports else "None", len(entries))
            if reports is None:
                reports = []
            while len(reports) < len(entries):
                idx = len(reports)
                reports.append(cls._fallback_report(entries[idx]["label"], entries[idx].get("detections")))

        for i, r in enumerate(reports):
            r["image_label"] = entries[i]["label"]
        return reports

    @staticmethod
    def _parse_json_array(text: str) -> list[dict] | None:
        """Extract a JSON array from model response."""
        text = re.sub(r"```(?:json)?", "", text).strip()
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
        # Fallback: single object → wrap in list
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict):
                    return [obj]
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Extract JSON from model response, tolerating markdown fences."""
        text = re.sub(r"```(?:json)?", "", text).strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _fallback_report(image_label: str, detections: list[dict] | None) -> dict:
        """Return a minimal report when the model call fails."""
        regions = []
        if detections:
            regions = [
                {"location": d["class_name"], "damage_type": "detected", "severity": "Unknown"}
                for d in detections
            ]
        return {
            "image_label"          : image_label,
            "overall_severity"     : "None",
            "regions"              : regions,
            "repair_category"      : "Pending assessment",
            "claim_recommendation" : "Investigate",
            "notes"                : "Automated assessment unavailable — manual review needed.",
        }


# ---------------------------------------------------------------------------
# Top-level function used by Flask
# ---------------------------------------------------------------------------
def assess_damage(image_bytes: bytes, image_label: str = "Vehicle") -> dict:
    """
    Full pipeline:  YOLO (if available) → Qwen VL report.

    Returns:
        {
            "annotated_image_bytes": <bytes or None>,
            "report": { ... }
        }
    """
    annotated_bytes = None
    detections      = None
    timings = {"yolo_infer": None, "yolo_annotate": None}

    if YOLODamageDetector.is_available():
        try:
            t0 = time.perf_counter()
            annotated_bytes, detections = YOLODamageDetector.detect(image_bytes)
            timings["yolo_infer"] = round((time.perf_counter() - t0) * 1000)
        except Exception as exc:
            logger.exception("YOLO detection failed, continuing with Qwen only: %s", exc)

    t1 = time.perf_counter()
    report = QwenVLAssessor.assess(image_bytes, annotated_bytes, detections, image_label)
    timings["qwen_total"] = round((time.perf_counter() - t1) * 1000)
    if isinstance(report, dict):
        report.setdefault("_timings_ms", {})
        report["_timings_ms"].update(timings)

    return {
        "annotated_image_bytes": annotated_bytes,
        "report"               : report,
    }


def assess_damage_batch(images: list[dict]) -> list[dict]:
    """
    Full pipeline for ALL images at once: YOLO per-image (fast) → single Gemma batch call.
    Args: list of {"image_bytes": bytes, "image_label": str}
    Returns: list of {"annotated_image_bytes": bytes|None, "report": dict}
    """
    yolo_available = YOLODamageDetector.is_available()
    entries = []
    for img in images:
        annotated_bytes = None
        detections = None
        if yolo_available:
            try:
                annotated_bytes, detections = YOLODamageDetector.detect(img["image_bytes"])
            except Exception as exc:
                logger.exception("YOLO failed for %s: %s", img["image_label"], exc)
        entries.append({
            "label"          : img["image_label"],
            "original_bytes" : img["image_bytes"],
            "detections"     : detections,
            "annotated_bytes": annotated_bytes,
        })

    reports = QwenVLAssessor.assess_batch(entries)

    results = []
    for i, entry in enumerate(entries):
        results.append({
            "annotated_image_bytes": entry["annotated_bytes"],
            "report"               : reports[i],
        })
    return results


def worst_severity(reports: list[dict]) -> str:
    """Return the highest severity across all per-image reports."""
    best = "None"
    for r in reports:
        sev = r.get("overall_severity", "None")
        if _SEVERITY_RANK.get(sev, 0) > _SEVERITY_RANK.get(best, 0):
            best = sev
    return best


def overall_recommendation(reports: list[dict]) -> str:
    """
    Roll up per-image recommendations.
    Reject > Investigate > Approve
    """
    rank = {"Approve": 0, "Investigate": 1, "Reject": 2}
    best = "Approve"
    for r in reports:
        rec = r.get("claim_recommendation", "Approve")
        if rank.get(rec, 0) > rank.get(best, 0):
            best = rec
    return best
