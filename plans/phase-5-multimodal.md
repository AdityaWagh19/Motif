# Phase 5 — Multimodal Ingestion

> **Status:** Not started  
> **Prerequisite:** Phase 4 complete (≥ 85% faithfulness on text corpus)  
> **Model downloads required:** whisper.cpp model (75–244 MB), PaddleOCR (~200 MB), moondream2 (T3 opt-in, ~900 MB)  
> **Estimated scope:** 3 files created, 3 files modified, ~700 lines of implementation

---

## Objective

Extend the ingestion pipeline to support images (OCR + optional captioning) and
audio (whisper.cpp transcription). DOCX with scanned pages uses OCR too.
PDF pages that previously returned empty text (scanned) are now processed.

By the end of this phase, every file type in the acceptance matrix is ingestible.

**Supported formats after Phase 5:**

| Format | Parser | T1 | T2 | T3 |
|---|---|---|---|---|
| .pdf (text) | PDFParser | ✅ | ✅ | ✅ |
| .pdf (scanned) | PDFParser + PaddleOCR | — | ✅ | ✅ |
| .docx | DOCXParser | ✅ | ✅ | ✅ |
| .md / .txt | MarkdownParser | ✅ | ✅ | ✅ |
| .png / .jpg | ImageParser + PaddleOCR | — | ✅ | ✅ |
| .png / .jpg (dense) | ImageParser + moondream2 caption | — | — | ✅ opt-in |
| .mp3 / .wav / .m4a | AudioParser (whisper.cpp) | ✅ (tiny) | ✅ (tiny) | ✅ (small) |

---

## Scope

**In scope:**
- `rag/ingestion/parsers/image.py` — OCR + optional moondream2 caption
- `rag/ingestion/parsers/audio.py` — whisper.cpp transcription
- `rag/ingestion/parsers/pdf.py` — update to use OCR for scanned pages (T2/T3)
- `rag/ingestion/parsers/base.py` — update `get_parser()` for image/audio
- `rag/models/model_manager.py` — add `get_whisper()` and `get_captioner()` methods
- `tests/unit/test_image_parser.py`
- `tests/unit/test_audio_parser.py`
- `tests/integration/test_multimodal_ingestion.py`

**Out of scope:**
- Surya-OCR (academic PDF layout; complex install — defer to Phase 7)
- NOUGAT (academic formula parsing — Phase 7)
- Video ingestion (Phase 7)

---

## Model Download Requirement

```bash
# whisper.cpp models (included in `motif setup` already via setup_models.py)
# T1/T2: ggml-tiny-q5_1.bin (75 MB)   — already in models/
# T3:    ggml-small-q5_1.bin (244 MB)  — already in models/

# PaddleOCR — downloaded at first use via paddleocr's auto-download
# No manual step required. Model stored in ~/.paddleocr/

# moondream2 — T3 opt-in only
motif setup --tier T3 --captioning   # downloads moondream2 Q4 (~900 MB)
```

---

## File Specifications

### `rag/ingestion/parsers/image.py`

**Strategy:**
1. Tier check: if T1 → raise ValueError (image parsing requires T2+)
2. Load PaddleOCR → run OCR on the image file
3. Extract text from OCR result
4. If `config.parsers.use_moondream` and image has low text density:
   - Run moondream2 to generate a descriptive caption
   - Prepend caption to text: "Image caption: {caption}\n\nOCR text: {text}"
5. Return one ParsedPage with `is_ocr=True`, `has_image=True`

```python
SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]
LOW_TEXT_THRESHOLD = 50  # chars — below this, try captioning

class ImageParser:

    def __init__(self, config: "RAGConfig") -> None:
        self._config = config

    def parse(self, path: Path) -> List[ParsedPage]:
        from rag.config import RAGConfig
        if self._config.resolved_tier == "T1":
            raise ValueError(
                "Image parsing requires T2 or T3 (PaddleOCR needs GPU or sufficient RAM). "
                "T1 (CPU-only) does not support image ingestion."
            )

        text = self._run_ocr(path)
        caption = ""

        if (
            self._config.parsers.use_moondream
            and len(text) < LOW_TEXT_THRESHOLD
        ):
            caption = self._run_caption(path)

        if caption and text:
            full_text = f"Image caption: {caption}\n\nOCR text: {text}"
        elif caption:
            full_text = f"Image caption: {caption}"
        else:
            full_text = text

        if not full_text.strip():
            return []

        return [ParsedPage(
            text=full_text.strip(),
            is_ocr=True,
            has_image=True,
        )]

    def _run_ocr(self, path: Path) -> str:
        """Run PaddleOCR on the image. Returns extracted text."""
        from paddleocr import PaddleOCR
        # Lazy-load PaddleOCR (slow first-time init — downloads models if missing)
        if not hasattr(self, "_ocr"):
            self._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        result = self._ocr.ocr(str(path), cls=True)
        if not result or not result[0]:
            return ""
        lines = []
        for line in result[0]:
            text_confidence = line[1]
            text = text_confidence[0]
            confidence = text_confidence[1]
            if confidence >= 0.6:  # drop low-confidence OCR lines
                lines.append(text)
        return " ".join(lines)

    def _run_caption(self, path: Path) -> str:
        """Run moondream2 to generate an image description."""
        try:
            captioner = get_model_manager().get_captioner(self._config)
            return captioner.caption(path)
        except Exception:
            return ""
```

**Add `get_captioner()` to ModelManager:**

```python
def get_captioner(self, config: RAGConfig) -> "Captioner":
    """Lazy-load moondream2 captioning model."""
    from rag.models.captioner import Captioner
    if self._captioner is None:
        model_path = Path(config.models.__dict__.get("captioner", "models/moondream2")).resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Captioning model not found: {model_path}\n"
                f"Run `motif setup --tier T3 --captioning` to download."
            )
        self._captioner = Captioner(model_path)
        self._captioner._load()
    return self._captioner
```

**`rag/models/captioner.py`** (new file for moondream2 wrapper):

```python
class Captioner:
    """moondream2 GGUF/ONNX wrapper for image captioning."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._model = None

    def _load(self) -> None:
        # moondream2 loaded via transformers pipeline or llama.cpp mmproj
        # Use transformers for simplicity:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir), trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self._model_dir),
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self._model.eval()

    def caption(self, image_path: Path) -> str:
        """Generate a one-sentence description of the image."""
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        enc_image = self._model.encode_image(img)
        result = self._model.answer_question(enc_image, "Describe this image in one sentence.", self._tokenizer)
        return result.strip()

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        del self._model
        self._model = None
        import gc; gc.collect()
```

---

### `rag/ingestion/parsers/audio.py`

**Backend:** `pywhispercpp` (Python bindings for whisper.cpp)

**Model selection:**
- T1/T2: `ggml-tiny-q5_1.bin` (fast, ~75 MB)
- T3: `ggml-small-q5_1.bin` (better accuracy, ~244 MB)

```python
SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]

class AudioParser:

    def __init__(self, config: "RAGConfig") -> None:
        self._config = config

    def parse(self, path: Path) -> List[ParsedPage]:
        """
        Transcribe audio file using whisper.cpp.

        Returns one ParsedPage per whisper segment group (~30-60 seconds each).
        Each ParsedPage carries start_time and end_time from whisper timestamps.
        """
        model_path = self._get_whisper_model_path()
        segments = self._transcribe(path, model_path)

        if not segments:
            return []

        # Group segments into ~512-token chunks by time window
        return self._group_segments(segments)

    def _get_whisper_model_path(self) -> Path:
        cfg = self._config
        whisper_model = cfg.models.whisper
        path = Path(whisper_model)
        if not path.is_absolute():
            # Relative to project root
            path = Path(cfg.models.llm_path).parent.parent / whisper_model
        if not path.exists():
            raise FileNotFoundError(
                f"Whisper model not found: {path}\n"
                f"Run `motif setup` to download it."
            )
        return path

    def _transcribe(self, audio_path: Path, model_path: Path) -> List[dict]:
        """
        Run whisper.cpp transcription. Returns list of segment dicts:
        {"text": str, "start": float, "end": float}
        """
        from pywhispercpp.model import Model
        model = Model(str(model_path), n_threads=self._config.llm.threads)
        segments = model.transcribe(str(audio_path))
        return [
            {
                "text": seg.text.strip(),
                "start": seg.t0 / 100.0,   # pywhispercpp times in centiseconds
                "end": seg.t1 / 100.0,
            }
            for seg in segments
            if seg.text.strip()
        ]

    def _group_segments(self, segments: List[dict]) -> List[ParsedPage]:
        """
        Group whisper segments into chunks of approximately TARGET_WORDS words.
        Each group becomes one ParsedPage with start_time/end_time set.
        """
        TARGET_WORDS = 350  # ≈ 512 tokens at 0.75 words/token ratio
        pages = []
        current_texts = []
        current_word_count = 0
        chunk_start_time = segments[0]["start"]

        for seg in segments:
            words = seg["text"].split()
            if current_word_count + len(words) > TARGET_WORDS and current_texts:
                pages.append(ParsedPage(
                    text=" ".join(current_texts),
                    start_time=chunk_start_time,
                    end_time=segments[segments.index(seg) - 1]["end"],
                    is_ocr=False,
                ))
                current_texts = [seg["text"]]
                current_word_count = len(words)
                chunk_start_time = seg["start"]
            else:
                current_texts.append(seg["text"])
                current_word_count += len(words)

        if current_texts:
            pages.append(ParsedPage(
                text=" ".join(current_texts),
                start_time=chunk_start_time,
                end_time=segments[-1]["end"],
            ))

        return pages
```

---

### Update `rag/ingestion/parsers/pdf.py` — OCR for scanned pages

In the parse loop, when `text` is empty (scanned page):

```python
if not text:
    if config.resolved_tier in ("T2", "T3"):
        # Run OCR via PaddleOCR
        ocr_text = self._ocr_page(page, doc_path)
        if ocr_text:
            pages.append(ParsedPage(
                text=ocr_text,
                page=page_num,
                is_ocr=True,
                has_image=True,
            ))
    # T1: skip scanned pages (no OCR)
    continue

def _ocr_page(self, fitz_page, doc_path: Path) -> str:
    """Export page as PNG and run PaddleOCR."""
    import tempfile, os
    mat = fitz.Matrix(2.0, 2.0)  # 2x resolution for better OCR
    pix = fitz_page.get_pixmap(matrix=mat)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pix.save(tmp.name)
        tmp_path = tmp.name
    try:
        from paddleocr import PaddleOCR
        if not hasattr(self, "_ocr"):
            self._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        result = self._ocr.ocr(tmp_path, cls=True)
        if not result or not result[0]:
            return ""
        return " ".join(line[1][0] for line in result[0] if line[1][1] >= 0.6)
    finally:
        os.unlink(tmp_path)
```

**Note:** `PDFParser.__init__` must now accept an optional `config` parameter.
Update all call sites in `ingest_path()` to pass `config`.

---

### Update `rag/ingestion/parsers/base.py` — `get_parser()`

```python
def get_parser(path: Path, config: "RAGConfig | None" = None) -> BaseParser:
    from rag.ingestion.parsers.pdf import PDFParser
    from rag.ingestion.parsers.docx import DOCXParser
    from rag.ingestion.parsers.markdown import MarkdownParser
    from rag.ingestion.parsers.image import ImageParser
    from rag.ingestion.parsers.audio import AudioParser

    ext = path.suffix.lower()

    if ext == ".pdf":
        return PDFParser(config)
    if ext == ".docx":
        return DOCXParser()
    if ext in MarkdownParser.SUPPORTED_EXTENSIONS:
        return MarkdownParser()
    if ext in ImageParser.SUPPORTED_EXTENSIONS:
        if config is None:
            raise ValueError("config required for ImageParser")
        return ImageParser(config)
    if ext in AudioParser.SUPPORTED_EXTENSIONS:
        if config is None:
            raise ValueError("config required for AudioParser")
        return AudioParser(config)

    raise ValueError(
        f"No parser for '{path.suffix}'. "
        f"Supported: .pdf, .docx, .md, .txt, .png, .jpg, .mp3, .wav, .m4a"
    )
```

**Update `ingest_path()` call site** to pass `config` to `get_parser()`:

```python
parser = get_parser(file, config)
```

---

## Test Specifications

### `tests/unit/test_image_parser.py`

```
test_image_parser_t1_raises:
    config.resolved_tier = "T1"
    parser = ImageParser(config)
    with pytest.raises(ValueError, match="T2 or T3"):
        parser.parse(Path("image.png"))

test_image_parser_extension_check:
    assert ImageParser.can_parse(Path("photo.jpg")) is True
    assert ImageParser.can_parse(Path("doc.pdf")) is False

test_image_parser_ocr_result_structure:
    # Mock PaddleOCR to return a fixed result
    # Verify parse() returns List[ParsedPage] with is_ocr=True
    [mock test — verify structure without real OCR model]
```

### `tests/unit/test_audio_parser.py`

```
test_audio_parser_extension_check:
    assert AudioParser.can_parse(Path("talk.mp3")) is True
    assert AudioParser.can_parse(Path("doc.pdf")) is False

test_audio_parser_group_segments_basic:
    # Test _group_segments directly with mock segment list
    segments = [
        {"text": "Hello world", "start": 0.0, "end": 3.0},
        {"text": "This is a test", "start": 3.0, "end": 6.0},
    ]
    parser = AudioParser(minimal_config)
    pages = parser._group_segments(segments)
    assert len(pages) >= 1
    assert pages[0].start_time == 0.0
    assert pages[-1].end_time == 6.0
    assert "Hello world" in pages[0].text

test_audio_parser_timestamps_in_page:
    # Verify ParsedPage.start_time and end_time are set
    pages = parser._group_segments([{"text": "x", "start": 10.5, "end": 15.3}])
    assert pages[0].start_time == 10.5
    assert pages[0].end_time == 15.3
```

### `tests/integration/test_multimodal_ingestion.py`

```python
@pytest.mark.slow
def test_audio_ingest_has_timestamps(minimal_config, tmp_path):
    # Generate a short wav file with pydub or use a fixture wav
    audio_file = tmp_path / "test.wav"
    # Write a minimal WAV header (or use a real fixture file from tests/fixtures/)
    # ...
    result = ingest_path(audio_file, config=minimal_config, recursive=False, console=None)
    assert result.files_processed == 1
    store = ChunkStore(minimal_config)
    chunks = store.fetch_batch([c.id for c in ... ])
    # Verify that at least one chunk has start_time set
    assert any(c.start_time is not None for c in chunks)

@pytest.mark.slow
def test_image_ingest_t2(t2_config, tmp_path):
    # Requires T2 config (or mock OCR)
    img_path = tmp_path / "test.png"
    _create_test_image_with_text(img_path, "This is test text for OCR")
    result = ingest_path(img_path, config=t2_config, recursive=False, console=None)
    assert result.files_processed == 1
    store = ChunkStore(t2_config)
    assert store.count() >= 1
```

---

## Validation Checklist

```bash
# 1. Imports
python -c "from rag.ingestion.parsers.image import ImageParser; print('ImageParser OK')"
python -c "from rag.ingestion.parsers.audio import AudioParser; print('AudioParser OK')"

# 2. Unit tests (no real models needed for most)
pytest tests/unit/test_image_parser.py tests/unit/test_audio_parser.py -v

# 3. Full multimodal integration test
pytest tests/integration/test_multimodal_ingestion.py -v -m slow

# 4. Functional check — ingest an audio file (WAV or MP3)
# motif
# /ingest ./test_audio.mp3
# /status → Documents: 1, Chunks: N
# What does the audio discuss?
# Expected: answer with timestamps [1] test_audio.mp3 @ 00:00–01:23

# 5. Accuracy within 5% of text-only baseline (manual check)
# Run the same 20 eval questions from Phase 4 on a mixed corpus
# (text + audio/image ingested)
# Accuracy should not degrade by more than 5% vs text-only baseline
```

---

## Post-Phase Documentation Updates

**`project-context/progress.md`:**
- Mark all Phase 3 (Multimodal) tasks ✅
- Update Phase Status Overview: Phase 3 → ✅ Done
- Add Metrics Snapshot: accuracy within 5% of baseline on mixed corpus

**`project-context/tests.md`:**
- Mark AUD-01, AUD-02 (audio transcription, timestamps) ✅
- Mark IMG-01, IMG-02 (image OCR, captioning gate) ✅
- Mark ING-20, ING-21 (scanned PDF OCR) ✅

**Update `config.template.toml`:**
- Add `[parsers]` section doc comment noting `use_moondream = true` requires
  `motif setup --tier T3 --captioning`
