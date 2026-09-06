import argparse
import gc
import html
import importlib.util
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BAD = "БАД"
FIRE = "Легковоспламеняющиеся"

SHARED_MODELS = Path(os.environ.get("SHARED_MODELS_PATH", "/shared_models"))
QWEN_MODEL_PATH = SHARED_MODELS / "Qwen" / "Qwen3.5-4B"
OCR_MODEL_PATH = SHARED_MODELS / "PaddlePaddle" / "PaddleOCR-VL-1.5"

ROOT = Path(__file__).resolve().parent
BAD_ADAPTER = ROOT / "adapters" / "bad"
FIRE_ADAPTER = ROOT / "adapters" / "fire"

MAX_IMAGES = 5
OCR_SHEET_SIZE = 1024
QWEN_SHEET_SIZE = 576
OCR_MAX_PIXELS = 1003520
OCR_MIN_PIXELS = 112896
OCR_MAX_NEW_TOKENS = 160
OCR_BATCH_SIZE = 32
QWEN_BATCH_SIZE = 24
MAX_DESCRIPTION_CHARS = 2200
MAX_OCR_CHARS = 2200
CPU_WORKERS = min(16, max(4, os.cpu_count() or 4))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

BAD_RULES = """Правила БАД:
- относится, если есть прямое указание БАД / биологически активная добавка / dietary supplement;
- спортивное питание при прямом указании на спортпит не относится;
- если явно сказано, что товар не является БАД, он не относится;
- без маркировки БАД / dietary supplement товар не относится."""

FIRE_RULES = """Правила Легковоспламеняющиеся:
- относится: самостоятельный источник воспламенения; содержит горючее вещество/ЛВЖ/горючий газ; опасный товар входит в комплект;
- не относится: устройство лишь используется с огнем/топливом, но не содержит его;
- не относится: горючее содержимое отсутствует в поставке;
- не относится: источник воспламенения встроен;
- не относится: горючий материал только компонент;
- не относится: опасный предмет не входит в комплект."""

TRIM_MARKER = "\n...[середина сокращена]...\n"
MIN_VISIBLE_ALPHA = 0.42
MAX_TRACE_CANDIDATES = 120
MIN_STATE_RULE_SIMILARITY = 0.12
MIN_STATE_RULE_GAP = 0.005
STATE_DECISION_WEIGHT = 0.025
STATE_DIRECTION_WEIGHT = 0.015

_BAD_TERM = r"(?:бад(?:ом|ами)?|биологическ\w*\s+активн\w*\s+добавк\w*|dietary\s+supplement)"
_DIRECT_NOT_BAD = re.compile(
    rf"(?:\bне\s+(?:явля\w*|счита\w*|относ\w*)\s+(?:к\s+)?{_BAD_TERM}\b|\b(?:это|товар|продукт)\s+не\s+{_BAD_TERM}\b|\bне\s+{_BAD_TERM}\b|\bnot\s+(?:an?\s+)?dietary\s+supplement\b)",
    re.I,
)
_COORDINATED_NOT_BAD = re.compile(
    r"\bне\s+явля\w*\s+(?:лекар\w+(?:\s+средств\w*)?)\s*,?\s*(?:и|или)\s+(?:не\s+явля\w*\s+)?бад\b",
    re.I,
)
_SPORT_DIRECT = re.compile(
    r"\b(?:спортивн\w*\s+(?:питан\w*|добавк\w*)|спортпит\w*|sports?\s+nutrition)\b", re.I
)
_SPORT_REFERENCE = re.compile(
    r"\b(?:част\w*\s+(?:систем\w*|комплекс\w*)|дополн\w*|совмещ\w*|применя\w*|производств\w*|рын\w*|мир\w*)\b[^.!?;]{0,80}\b(?:спортивн\w*\s+питан\w*|спортпит\w*)\b|\b(?:включ\w*|добав\w*)\b[^.!?;]{0,80}\b(?:в|к)\s+(?:свой\w*\s+)?(?:систем\w*|комплекс\w*|рацион\w*)\s+спортивн\w*\s+питан\w*\b|\bспортивн\w*\s+питан\w*\s+и\s+(?:косметик\w*|красот\w*)\b|\bспортивн\w*\s+питан\w*\b[^.!?;]{0,70}\b(?:необходим\w*\s+элемент\w*|част\w*)\s+рацион\w*\b",
    re.I,
)
_SPORT_PRODUCT = re.compile(
    r"\b(?:bcaa|бцаа|l[-\s]?карнитин\w*|л[-\s]?карнитин\w*|левокарнитин\w*|протеин(?:овый|овая|овые|а)?|protein|аминокислотн\w*\s+(?:комплекс\w*|смес\w*|добавк\w*))\b",
    re.I,
)
_SPORT_CONTEXT = re.compile(
    r"\b(?:спорт\w*|атлет\w*|трениров\w*|мышц\w*|жиросжиг\w*|предтрен\w*|посттрен\w*)\b", re.I
)
_BAD_LONG = re.compile(
    r"\b(?:биологическ\w*\s+активн\w*\s+добавк\w*|dietary\s+supplement)\b", re.I
)
_BAD_SHORT = re.compile(r"\bбад\b", re.I)
_BAD_REFERENCE = re.compile(
    r"\b(?:для|о|об|рынок|рынка|производств\w*|категори\w*|каталог\w*|магазин\w*|упаковк\w*|контейнер\w*|органайзер\w*|таблетниц\w*|маркировк\w*)\s+(?:для\s+)?бад\b|\bдля\b.{0,30}\bбад\b|\bмежду\b.{0,80}\b(?:бад|биологическ\w*\s+активн\w*\s+добавк\w*)\b|\b(?:схож\w*|сравн\w*)\b.{0,45}\b(?:с\s+)?бад\b|\b(?:употребля\w*|принима\w*|хранен\w*)\b.{0,80}\b(?:бад|биологическ\w*\s+активн\w*\s+добавк\w*)\b|\b(?:лекарств\w*|маз\w*|витамин\w*)\b(?:[^.!?;]{0,55}\s(?:и|или|,))[^.!?;]{0,35}\bбад\b|\bбад\b.{0,35}\b(?:вообще|часто)\b",
    re.I,
)
_PHOTO_MARKER = re.compile(r"\[Фото\s+(\d+)\]", re.I)


def norm_id(x):
    try:
        f = float(x)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return str(x)


def sort_key(p):
    try:
        return (0, int(p.stem))
    except Exception:
        return (1, p.name)


def get_image_paths(images_root, pid):
    folder = images_root / norm_id(pid)
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
        key=sort_key,
    )[:MAX_IMAGES]


def fit_tile(img, box):
    img = img.convert("RGB")
    img.thumbnail(box, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", box, "white")
    canvas.paste(img, ((box[0] - img.width) // 2, (box[1] - img.height) // 2))
    return canvas


def make_sheet_from_paths(paths, size):
    if not paths:
        return Image.new("RGB", (size, size), "white")

    n = len(paths)
    if n == 1:
        cols, rows = 1, 1
    elif n <= 4:
        cols, rows = 2, math.ceil(n / 2)
    else:
        cols, rows = 3, 2

    gap = max(4, size // 180)
    tw = (size - gap * (cols - 1)) // cols
    th = (size - gap * (rows - 1)) // rows
    sheet = Image.new("RGB", (size, size), "white")

    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                tile = fit_tile(im, (tw, th))
        except Exception:
            tile = Image.new("RGB", (tw, th), "white")
        sheet.paste(tile, ((i % cols) * (tw + gap), (i // cols) * (th + gap)))
    return sheet


def trim_text(s, n, ratio=0.70):
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    if len(s) <= n:
        return s
    h = int(n * ratio)
    return s[:h] + "\n...[середина сокращена]...\n" + s[-(n - h):]


def clean_ocr(s):
    s = "" if s is None else str(s)
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _plain(value):
    value = html.unescape(re.sub(r"<[^>]+>", " ", "" if value is None else str(value)))
    return re.sub(r"\s+", " ", value).strip()


def _bad_pieces(value, source):
    raw = "" if value is None else str(value)
    blocks = []
    if source == "ocr":
        marks = list(_PHOTO_MARKER.finditer(raw))
        if marks:
            for index, mark in enumerate(marks):
                end = marks[index + 1].start() if index + 1 < len(marks) else len(raw)
                blocks.append((_plain(raw[mark.end():end]), int(mark.group(1))))
        else:
            blocks.append((_plain(raw), None))
    else:
        blocks.append((_plain(raw), None))
    for text, photo_number in blocks:
        if not text:
            continue
        sentences = [part.strip() for part in re.split(r"(?<=[.!?;])\s+|\s*[•·]\s*", text) if part.strip()]
        for sentence in sentences or [text]:
            words = sentence.split()
            if len(sentence) <= 210:
                yield sentence, photo_number
                continue
            for start in range(0, len(words), 24):
                piece = " ".join(words[start:start + 34])
                if piece:
                    yield piece, photo_number
                if start + 34 >= len(words):
                    break


def _near_sport_context(text, radius=75):
    return any(
        _SPORT_CONTEXT.search(text[max(0, match.start() - radius):min(len(text), match.end() + radius)])
        for match in _SPORT_PRODUCT.finditer(text)
    )


def _bad_rule_match(text, source):
    negative = None if source == "name" else (_DIRECT_NOT_BAD.search(text) or _COORDINATED_NOT_BAD.search(text))
    if negative:
        return "explicit_not_bad", 100, negative
    sport = _SPORT_DIRECT.search(text)
    if sport and (source == "name" or (len(text) >= 35 and not _SPORT_REFERENCE.search(text))):
        return "sports_nutrition", 90, sport
    product = _SPORT_PRODUCT.search(text)
    if product and (source == "name" or _near_sport_context(text)):
        return "sports_nutrition", 80, product
    marking = None if source == "name" else _BAD_LONG.search(text)
    if marking and not _BAD_REFERENCE.search(text):
        return "bad_marking", 70, marking
    marking = None if source == "name" else _BAD_SHORT.search(text)
    if marking and not _BAD_REFERENCE.search(text):
        return "bad_marking", 60, marking
    return None


def _bad_candidate(source, text, photo_number, rule, priority, match):
    if len(text) > 165:
        left = max(0, match.start() - 70)
        right = min(len(text), match.end() + 90)
        quote = text[left:right].strip(" ,;:-")
        quote = ("…" if left else "") + quote + ("…" if right < len(text) else "")
    else:
        quote = text
    return {
        "source": source,
        "display_fragment": quote,
        "char_start": match.start(),
        "photo_number": photo_number,
        "regex_rule": rule,
        "regex_priority": priority,
    }


def select_bad_evidence(row):
    matches = []
    for source, value in (("name", row.get("name", "")), ("description", row.get("description", "")), ("ocr", row.get("ocr_text", ""))):
        for text, photo_number in _bad_pieces(value, source):
            found = _bad_rule_match(text, source)
            if found:
                matches.append(_bad_candidate(source, text, photo_number, *found))
    if not matches:
        return None
    bonus = {"name": 3, "description": 2, "ocr": 1}
    matches.sort(key=lambda item: (-item["regex_priority"], -bonus[item["source"]], len(item["display_fragment"]), item["char_start"]))
    top = matches[0]
    modality = lambda item: "image" if item["source"] == "ocr" else "card"
    sources = {modality(top)} | {
        modality(item) for item in matches[1:]
        if item["regex_rule"] == top["regex_rule"] and modality(item) != modality(top)
    }
    top["evidence_sources"] = "+".join(sorted(sources))
    return top


class MappedHTML(HTMLParser):
    def __init__(self, raw):
        super().__init__(convert_charrefs=False)
        self.raw = str(raw)
        self.starts = [0] + [i + 1 for i, char in enumerate(self.raw) if char == "\n"]
        self.chars, self.mapping = [], []

    def position(self):
        line, column = self.getpos()
        return self.starts[line - 1] + column

    def append(self, text, start, end=None):
        self.chars.extend(text)
        self.mapping.extend([(start, end)] * len(text) if end is not None else [(start + i, start + i + 1) for i in range(len(text))])

    def handle_data(self, data):
        self.append(data, self.position())

    def handle_entityref(self, name):
        start = self.position()
        end = start + len(name) + 1
        if self.raw[end:end + 1] == ";":
            end += 1
        self.append(html.unescape(self.raw[start:end]), start, end)

    def handle_charref(self, name):
        start = self.position()
        end = start + len(name) + 2
        if self.raw[end:end + 1] == ";":
            end += 1
        self.append(html.unescape(self.raw[start:end]), start, end)

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"p", "div", "li", "ul", "ol", "br", "h1", "h2", "h3", "tr"}:
            self.append("\n", self.position(), self.position())

    def handle_endtag(self, tag):
        self.handle_starttag(tag, [])


def visible_text(raw):
    parser = MappedHTML(raw)
    parser.feed(str(raw))
    parser.close()
    return "".join(parser.chars), parser.mapping


def sentence_windows(text, max_chars=165):
    boundaries = [0] + [m.end() for m in re.finditer(r"[!?;]+\s+|(?<!\d)\.(?!\d)\s+|\n\s*\n", text)] + [len(text)]
    for left, right in zip(boundaries, boundaries[1:]):
        words = list(re.finditer(r"\S+", text[left:right]))
        if not words:
            continue
        if len(" ".join(text[left:right].split())) <= max_chars:
            yield left + words[0].start(), left + words[-1].end(), True
            continue
        start = 0
        while start < len(words):
            end = start
            while end + 1 < len(words) and len(" ".join(text[left + words[start].start():left + words[end + 1].end()].split())) <= max_chars:
                end += 1
            yield left + words[start].start(), left + words[end].end(), False
            if end == len(words) - 1:
                break
            start = max(start + 1, end - max(1, (end - start) // 3))


def photo_parts(value):
    matches = list(_PHOTO_MARKER.finditer(value))
    if not matches:
        yield 0, len(value), None
        return
    if value[:matches[0].start()].strip():
        yield 0, matches[0].start(), None
    for index, match in enumerate(matches):
        yield match.end(), matches[index + 1].start() if index + 1 < len(matches) else len(value), int(match.group(1))


def attn_impl():
    return "flash_attention_2" if importlib.util.find_spec("flash_attn") is not None else "sdpa"


def load_with_attention(model_cls, model_path, **kwargs):
    preferred = attn_impl()
    try:
        print(f"Loading {model_path} with {preferred}", flush=True)
        return model_cls.from_pretrained(
            str(model_path),
            attn_implementation=preferred,
            local_files_only=True,
            trust_remote_code=True,
            **kwargs,
        )
    except Exception as e:
        if preferred == "sdpa":
            raise
        print(f"flash_attention_2 unavailable at runtime ({type(e).__name__}); fallback to SDPA", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        return model_cls.from_pretrained(
            str(model_path),
            attn_implementation="sdpa",
            local_files_only=True,
            trust_remote_code=True,
            **kwargs,
        )


def run_ocr(df, images_root, max_stage_seconds=None):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText
        ocr_cls = AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForCausalLM
        ocr_cls = AutoModelForCausalLM

    assert OCR_MODEL_PATH.exists(), f"Missing shared OCR model: {OCR_MODEL_PATH}"

    processor = AutoProcessor.from_pretrained(
        str(OCR_MODEL_PATH),
        local_files_only=True,
        trust_remote_code=True,
    )
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    try:
        processor.image_processor.max_pixels = OCR_MAX_PIXELS
        processor.image_processor.min_pixels = OCR_MIN_PIXELS
    except Exception:
        pass

    model = load_with_attention(
        ocr_cls,
        OCR_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda:0").eval()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "OCR:"},
        ],
    }]
    ocr_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_paths = [get_image_paths(images_root, pid) for pid in df["id"]]
    ocr_texts = [""] * len(df)
    active = [i for i, paths in enumerate(image_paths) if paths]

    print(
        f"OCR: {len(active)}/{len(df)} products have images | "
        f"one {OCR_SHEET_SIZE}x{OCR_SHEET_SIZE} contact sheet per product",
        flush=True,
    )
    if not active:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
        return ocr_texts

    pool = ThreadPoolExecutor(max_workers=CPU_WORKERS)
    bs = OCR_BATCH_SIZE
    pos = 0
    started = time.time()

    while pos < len(active):
        cur = min(bs, len(active) - pos)
        idxs = active[pos:pos + cur]
        try:
            sheets = list(pool.map(
                lambda j: make_sheet_from_paths(image_paths[j], OCR_SHEET_SIZE),
                idxs,
            ))
            prompts = [ocr_prompt] * len(sheets)
            inputs = processor(
                text=prompts,
                images=sheets,
                padding=True,
                return_tensors="pt",
            )
            inputs = {
                k: v.to("cuda:0", non_blocking=True)
                for k, v in inputs.items()
                if isinstance(v, torch.Tensor)
            }

            input_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=OCR_MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )
            generated = outputs[:, input_len:]
            decoded = processor.batch_decode(generated, skip_special_tokens=True)

            for j, text in zip(idxs, decoded):
                ocr_texts[j] = clean_ocr(text)

            pos += len(idxs)
            elapsed = time.time() - started
            rate = pos / max(elapsed, 1e-6)
            eta = (len(active) - pos) / max(rate, 1e-6) / 60
            print(
                f"OCR {pos}/{len(active)} | batch={bs} | {rate:.2f} product/s | ETA {eta:.1f}m",
                flush=True,
            )

            del sheets, inputs, outputs, generated, decoded

            if max_stage_seconds is not None and (time.time() - started) >= max_stage_seconds:
                print(
                    f"OCR soft time limit reached after {pos}/{len(active)} products; "
                    "remaining products continue with empty OCR.",
                    flush=True,
                )
                break
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            bs //= 2
            if bs < 1:
                raise
            print(f"OCR OOM -> batch={bs}", flush=True)

    pool.shutdown(wait=True)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return ocr_texts


def build_qwen_prompt(row):
    category = str(row["category"])
    rules = BAD_RULES if category == BAD else FIRE_RULES
    name = "" if pd.isna(row.get("name")) else str(row.get("name"))
    desc = trim_text(row.get("description", ""), MAX_DESCRIPTION_CHARS)
    ocr = trim_text(row.get("ocr_text", ""), MAX_OCR_CHARS)
    ocr_block = ""
    if ocr.strip():
        ocr_block = (
            "\n\nТекст, автоматически распознанный на фотографиях товара (OCR).\n"
            "OCR может содержать ошибки, поэтому используй его только как дополнительный источник информации:\n\n"
            f"{ocr}\n"
        )

    return f"""Ты решаешь бинарную классификацию товара.

{rules}

Название:
{name}

Описание:
{desc}{ocr_block}

На изображении объединены все фотографии товара.

Предскажи целевую метку из обучающей разметки.
Ответь строго одним символом: 0 или 1.

Ответ:"""


def qwen_chat(processor, row):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": build_qwen_prompt(row)},
        ],
    }]
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def fire_prompt_parts(row):
    prompt = build_qwen_prompt(row)
    prefix = "Ты решаешь бинарную классификацию товара.\n\n"
    rule_start = len(prefix)
    rule_end = rule_start + len(FIRE_RULES)
    name = "" if pd.isna(row.get("name")) else str(row.get("name"))
    desc = trim_text(row.get("description", ""), MAX_DESCRIPTION_CHARS)
    ocr = trim_text(row.get("ocr_text", ""), MAX_OCR_CHARS)
    name_start = rule_end + len("\n\nНазвание:\n")
    desc_start = name_start + len(name) + len("\n\nОписание:\n")
    fields = [("name", name, name_start), ("description", desc, desc_start)]
    if ocr.strip():
        intro = "\n\nТекст, автоматически распознанный на фотографиях товара (OCR).\nOCR может содержать ошибки, поэтому используй его только как дополнительный источник информации:\n\n"
        fields.append(("ocr", ocr, desc_start + len(desc) + len(intro)))
    rules = []
    cursor = rule_start
    for line in FIRE_RULES.splitlines(keepends=True):
        plain = line.rstrip("\n")
        if plain.startswith("- "):
            rules.append({"rule_text": plain[2:], "char_start": cursor + 2, "char_end": cursor + len(plain)})
        cursor += len(line)
    return prompt, fields, rules


def fire_candidates(row):
    prompt, fields, rules = fire_prompt_parts(row)
    candidates = []
    for source, value, field_offset in fields:
        parts = photo_parts(value) if source == "ocr" else [(0, len(value), None)]
        for part_start, part_end, photo_number in parts:
            cursor = part_start
            for piece in value[part_start:part_end].split(TRIM_MARKER):
                readable, mapping = visible_text(piece)
                for start, end, complete in sentence_windows(readable):
                    display = " ".join(readable[start:end].split())
                    alpha = sum(char.isalpha() or char.isspace() for char in display) / max(1, len(display))
                    if not 12 <= len(display) <= 165 or alpha < MIN_VISIBLE_ALPHA or re.search(r"(?:https?://|www\.)", display, re.I):
                        continue
                    raw_start = cursor + mapping[start][0]
                    raw_end = cursor + mapping[end - 1][1]
                    raw = value[raw_start:raw_end]
                    if raw.strip() and TRIM_MARKER.strip() not in raw:
                        candidates.append({
                            "source": source,
                            "fragment": raw,
                            "display_fragment": display,
                            "char_start": field_offset + raw_start,
                            "char_end": field_offset + raw_end,
                            "photo_number": photo_number,
                            "complete_sentence": complete,
                            "alpha_fraction": alpha,
                        })
                cursor += len(piece) + len(TRIM_MARKER)
    unique = {(item["source"], item["char_start"], item["char_end"]): item for item in candidates}
    items = list(unique.values())
    if len(items) > MAX_TRACE_CANDIDATES:
        items = sorted(items, key=lambda item: (-item["complete_sentence"], -item["alpha_fraction"], item["char_start"]))[:MAX_TRACE_CANDIDATES]
    return prompt, items, rules


def fire_token_plan(processor, row, chat, actual_ids):
    prompt, segments, rules = fire_candidates(row)
    chat_start = chat.index(prompt)
    encoded = processor.tokenizer(chat, add_special_tokens=False, return_offsets_mapping=True)
    raw = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    vision_end = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    raw_end = raw.index(vision_end)
    actual_end = actual_ids.index(vision_end)
    if raw[raw_end:] != actual_ids[actual_end:]:
        raise RuntimeError("Processor/tokenizer suffix mismatch")
    shift = actual_end - raw_end

    def bind(item):
        start = chat_start + item["char_start"]
        end = chat_start + item["char_end"]
        before = max(index for index, (_, token_end) in enumerate(offsets) if 0 < token_end <= start)
        after = max(index for index, (_, token_end) in enumerate(offsets) if 0 < token_end <= end)
        if before <= raw_end or after <= before:
            return None
        return {**item, "token_before": before + shift, "token_after": after + shift}

    segments = [bound for item in segments if (bound := bind(item)) is not None]
    rules = [bound for item in rules if (bound := bind(item)) is not None]
    positions = sorted({len(actual_ids) - 1} | {position for item in segments + rules for position in (item["token_before"], item["token_after"])})
    lookup = {position: index for index, position in enumerate(positions)}
    for item in segments + rules:
        item["before_index"] = lookup[item["token_before"]]
        item["after_index"] = lookup[item["token_after"]]
    return segments, rules, positions


def select_fire_evidence(segments, rules, hidden, margins, pred, boundary, class_direction):
    if not segments or not rules:
        return None
    rule_vectors = []
    for rule in rules:
        vector = hidden[rule["after_index"]] - hidden[rule["before_index"]]
        norm = np.linalg.norm(vector)
        if norm > 1e-8:
            rule_vectors.append((rule, vector / norm))
    if not rule_vectors:
        return None
    direction = class_direction / max(np.linalg.norm(class_direction), 1e-8)
    trace = []
    absolute_deltas = []
    for item in segments:
        vector = hidden[item["after_index"]] - hidden[item["before_index"]]
        norm = np.linalg.norm(vector)
        if norm <= 1e-8:
            continue
        vector = vector / norm
        similarities = [float(vector @ rule_vector) for _, rule_vector in rule_vectors]
        best = int(np.argmax(similarities))
        before = float(margins[item["before_index"]])
        after = float(margins[item["after_index"]])
        delta = after - before
        absolute_deltas.append(abs(delta))
        trace.append({
            **item,
            "nearest_rule": rule_vectors[best][0]["rule_text"],
            "state_rule_similarity": similarities[best],
            "delta": delta,
            "signed_delta": (2 * pred - 1) * delta,
            "decision_alignment": float((vector @ direction) * (2 * pred - 1)),
        })
    if not trace:
        return None
    scale = max(float(np.median(absolute_deltas)), 1e-6)
    for item in trace:
        item["rank_score"] = item["state_rule_similarity"] + STATE_DECISION_WEIGHT * np.tanh(item["signed_delta"] / scale) + STATE_DIRECTION_WEIGHT * item["decision_alignment"] + 0.01 * float(item["complete_sentence"])
    ranked = sorted(trace, key=lambda item: (-item["rank_score"], item["char_start"]))
    top = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    gap = top["state_rule_similarity"] - (runner_up["state_rule_similarity"] if runner_up else 0.0)
    if top["state_rule_similarity"] < MIN_STATE_RULE_SIMILARITY or gap < MIN_STATE_RULE_GAP:
        return None
    modality = lambda item: "image" if item["source"] == "ocr" else "card"
    top_modality = modality(top)
    sources = {top_modality} | {
        modality(item) for item in ranked[1:]
        if modality(item) != top_modality
        and item["nearest_rule"] == top["nearest_rule"]
        and item["state_rule_similarity"] >= MIN_STATE_RULE_SIMILARITY
        and item["rank_score"] >= top["rank_score"] - 0.05
    }
    top["evidence_sources"] = "+".join(sorted(sources))
    return top


def evidence_comment(category, pred, row, top, has_images):
    verdict = "не бан" if int(pred) else "бан"
    if category == BAD and top is None:
        basis = "по карточке товара и изображению" if str(row.get("ocr_text", "")).strip() else "по карточке товара"
        return f"Вердикт: {verdict}. Вывод сделан {basis}. Прямой маркировки «БАД» или «dietary supplement» в доступном тексте не найдено."
    if top is None:
        basis = "по данным карточки и фотографий" if has_images else "по данным карточки"
        return f"Вердикт: {verdict}. Вывод сделан {basis}. Одну короткую цитату для пояснения выделить нельзя; карточку следует проверить целиком."
    source = top["source"]
    sources = set(top.get("evidence_sources", "image" if source == "ocr" else "card").split("+"))
    basis = "по изображению товара" if sources == {"image"} else "по карточке товара" if sources == {"card"} else "по карточке товара и изображению"
    location = {"name": "В названии", "description": "В описании", "ocr": "В тексте с фотографий"}[source]
    quote = top["display_fragment"].replace("<", "‹").replace(">", "›").replace("«", "“").replace("»", "”")
    comment = f"Вердикт: {verdict}. Вывод сделан {basis}. {location} указано: «{quote}»"
    if not quote.endswith((".", "!", "?")):
        comment += "."
    return comment


class FinalNormCapture:
    def __init__(self, norm):
        self.value = None
        self.handle = norm.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        self.value = output.detach()

    def close(self):
        self.handle.remove()


def run_qwen(df, images_root):
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3_5ForConditionalGeneration
        qwen_cls = Qwen3_5ForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForCausalLM
        qwen_cls = AutoModelForCausalLM

    wheel_dir = ROOT / "wheels"
    for whl in sorted(wheel_dir.glob("peft-*.whl")):
        if str(whl) not in sys.path:
            sys.path.insert(0, str(whl))

    import peft
    try:
        import peft.tuners.lora.torchao as peft_torchao
        peft_torchao.is_torchao_available = lambda: False
    except Exception:
        pass
    from peft import PeftModel

    assert QWEN_MODEL_PATH.exists(), f"Missing shared Qwen model: {QWEN_MODEL_PATH}"
    assert (BAD_ADAPTER / "adapter_config.json").exists(), BAD_ADAPTER
    assert (FIRE_ADAPTER / "adapter_config.json").exists(), FIRE_ADAPTER

    processor = AutoProcessor.from_pretrained(
        str(QWEN_MODEL_PATH),
        local_files_only=True,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    try:
        processor.image_processor.size["longest_edge"] = QWEN_SHEET_SIZE * QWEN_SHEET_SIZE
        processor.image_processor.size["shortest_edge"] = 224 * 224
    except Exception:
        pass

    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    assert len(zero) == 1 and len(one) == 1
    zero_id, one_id = zero[0], one[0]

    base = load_with_attention(
        qwen_cls,
        QWEN_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    base.tie_weights()
    base.to("cuda:0")
    base.eval()
    base.config.use_cache = False

    model = PeftModel.from_pretrained(
        base,
        str(BAD_ADAPTER),
        adapter_name="BAD",
        is_trainable=False,
        low_cpu_mem_usage=False,
    )
    model.load_adapter(
        str(FIRE_ADAPTER),
        adapter_name="FIRE",
        is_trainable=False,
        low_cpu_mem_usage=False,
    )
    model.eval()

    meta = [n for n, p in model.named_parameters() if getattr(p, "is_meta", False)]
    if meta:
        raise RuntimeError(f"Meta parameters remain after LoRA load: {meta[:20]}")

    norm = base.model.language_model.norm
    head = base.get_output_embeddings()
    label_weight = head.weight[[zero_id, one_id]].detach().float()
    label_bias = head.bias[[zero_id, one_id]].detach().float() if head.bias is not None else None
    class_direction = (label_weight[1] - label_weight[0]).cpu().numpy()

    image_paths = [get_image_paths(images_root, pid) for pid in df["id"]]
    preds = np.zeros(len(df), dtype=np.int8)
    p1s = np.zeros(len(df), dtype=np.float32)
    comments = [""] * len(df)

    pool = ThreadPoolExecutor(max_workers=CPU_WORKERS)

    for category, adapter in [(BAD, "BAD"), (FIRE, "FIRE")]:
        idxs = np.flatnonzero(df["category"].astype(str).values == category).tolist()
        if not idxs:
            continue
        model.set_adapter(adapter)
        bs = QWEN_BATCH_SIZE
        pos = 0
        started = time.time()
        capture = FinalNormCapture(norm) if category == FIRE else None

        while pos < len(idxs):
            cur = min(bs, len(idxs) - pos)
            batch_idxs = idxs[pos:pos + cur]
            try:
                sheets = list(pool.map(
                    lambda j: make_sheet_from_paths(image_paths[j], QWEN_SHEET_SIZE),
                    batch_idxs,
                ))
                records = [df.iloc[j].to_dict() for j in batch_idxs]
                texts = [qwen_chat(processor, r) for r in records]
                inputs = processor(
                    text=texts,
                    images=sheets,
                    padding=True,
                    return_tensors="pt",
                )
                actual_ids = inputs["input_ids"].tolist()
                plans = None
                if category == FIRE:
                    plans = list(pool.map(
                        lambda args: fire_token_plan(processor, args[0], args[1], args[2]),
                        zip(records, texts, actual_ids),
                    ))
                inputs = {
                    k: v.to("cuda:0", non_blocking=True)
                    for k, v in inputs.items()
                    if isinstance(v, torch.Tensor)
                }

                with torch.inference_mode():
                    out = model(**inputs, use_cache=False, logits_to_keep=1)
                    logits = out.logits[:, -1, [zero_id, one_id]].float()
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

                threshold = 0.12 if category == BAD else 0.5

                for j, p1 in zip(batch_idxs, probs):
                    p1s[j] = float(p1)
                    preds[j] = int(p1 >= threshold)

                if category == BAD:
                    tops = list(pool.map(select_bad_evidence, records))
                    for j, record, top in zip(batch_idxs, records, tops):
                        comments[j] = evidence_comment(BAD, preds[j], record, top, bool(image_paths[j]))
                else:
                    if capture.value is None or capture.value.shape[0] != len(batch_idxs):
                        raise RuntimeError("Final norm capture failed")
                    state_chunks = []
                    lengths = []
                    for local_index, plan in enumerate(plans):
                        positions = plan[2]
                        position_tensor = torch.tensor(positions, dtype=torch.long, device=capture.value.device)
                        state_chunks.append(capture.value[local_index].index_select(0, position_tensor).float())
                        lengths.append(len(positions))
                    packed_states = torch.cat(state_chunks, dim=0)
                    packed_margins = torch.nn.functional.linear(packed_states, label_weight, label_bias)
                    packed_states = packed_states.cpu().numpy()
                    packed_margins = (packed_margins[:, 1] - packed_margins[:, 0]).cpu().numpy()
                    offset = 0
                    for local_index, (j, record, plan) in enumerate(zip(batch_idxs, records, plans)):
                        segments, rules, positions = plan
                        end = offset + lengths[local_index]
                        top = select_fire_evidence(
                            segments,
                            rules,
                            packed_states[offset:end],
                            packed_margins[offset:end],
                            int(preds[j]),
                            0.0,
                            class_direction,
                        )
                        comments[j] = evidence_comment(FIRE, preds[j], record, top, bool(image_paths[j]))
                        offset = end
                    capture.value = None
                    del state_chunks, lengths, packed_states, packed_margins

                pos += len(batch_idxs)
                elapsed = time.time() - started
                rate = pos / max(elapsed, 1e-6)
                eta = (len(idxs) - pos) / max(rate, 1e-6) / 60
                print(
                    f"Qwen [{category}] {pos}/{len(idxs)} | batch={bs} | "
                    f"{rate:.2f} product/s | ETA {eta:.1f}m",
                    flush=True,
                )
                del sheets, records, texts, inputs, out, logits, probs, actual_ids, plans
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                bs //= 2
                if bs < 1:
                    raise
                print(f"Qwen OOM -> batch={bs}", flush=True)

        if capture is not None:
            capture.close()

    pool.shutdown(wait=True)
    del model, base, processor
    gc.collect()
    torch.cuda.empty_cache()
    if not all(comments):
        raise RuntimeError("Comment generation completeness check failed")
    return p1s, preds, comments


def format_result(comment, pred):
    if not 50 <= len(comment) <= 300:
        raise ValueError(f"Invalid comment length: {len(comment)}")
    verdict = "не бан" if int(pred) == 1 else "бан"
    return f"<комментарий>{comment}<вердикт>{verdict}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", "--test-data-path", "-i", dest="test_data_path", required=True)
    parser.add_argument("--output_path", "--output-path", "-o", dest="output_path", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("CUDA capability:", torch.cuda.get_device_capability(0), flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    data_path = Path(args.test_data_path)
    images_root = data_path.parent / "images"
    df = pd.read_csv(data_path).copy()

    required = {"id", "name", "description", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["category"] = df["category"].astype(str)
    print(f"Rows: {len(df)} | images root: {images_root}", flush=True)

    t0 = time.time()

    if len(df) <= 50:
        total_soft_budget = 150.0
    elif len(df) <= 2500:
        total_soft_budget = 1080.0
    else:
        total_soft_budget = 2280.0
    ocr_soft_budget = total_soft_budget * 0.55

    print(
        f"Soft runtime budget: {total_soft_budget / 60:.1f}m | "
        f"OCR share: {ocr_soft_budget / 60:.1f}m",
        flush=True,
    )

    try:
        df["ocr_text"] = run_ocr(df, images_root, max_stage_seconds=ocr_soft_budget)
    except Exception as e:
        print(
            f"OCR stage failed ({type(e).__name__}: {e}). "
            "Fallback: Qwen runs without OCR so the submission still produces all rows.",
            flush=True,
        )
        df["ocr_text"] = ""
        gc.collect()
        torch.cuda.empty_cache()

    print(f"OCR stage: {(time.time() - t0) / 60:.2f} min", flush=True)

    t1 = time.time()
    p1s, preds, comments = run_qwen(df, images_root)
    df["p1"] = p1s
    df["pred"] = preds
    print(f"Qwen stage: {(time.time() - t1) / 60:.2f} min", flush=True)

    df["result"] = [
        format_result(comment, pred)
        for comment, pred in zip(comments, df["pred"])
    ]

    out = df[["id", "result"]].copy()
    if len(out) != len(df) or out["id"].isna().any() or out["result"].isna().any():
        raise RuntimeError("Output completeness check failed")
    if not out["result"].str.match(r"^<комментарий>.{50,300}<вердикт>(бан|не бан)$").all():
        raise RuntimeError("Output format validation failed")

    out.to_csv(args.output_path, index=False)
    print(f"Saved: {args.output_path}", flush=True)
    print(f"Total: {(time.time() - t0) / 60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
