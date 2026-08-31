import argparse
import gc
import importlib.util
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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

    image_paths = [get_image_paths(images_root, pid) for pid in df["id"]]
    preds = np.zeros(len(df), dtype=np.int8)
    p1s = np.zeros(len(df), dtype=np.float32)

    pool = ThreadPoolExecutor(max_workers=CPU_WORKERS)

    for category, adapter in [(BAD, "BAD"), (FIRE, "FIRE")]:
        idxs = np.flatnonzero(df["category"].astype(str).values == category).tolist()
        if not idxs:
            continue
        model.set_adapter(adapter)
        bs = QWEN_BATCH_SIZE
        pos = 0
        started = time.time()

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
                inputs = {
                    k: v.to("cuda:0", non_blocking=True)
                    for k, v in inputs.items()
                    if isinstance(v, torch.Tensor)
                }

                with torch.inference_mode():
                    out = model(**inputs, use_cache=False, logits_to_keep=1)
                    logits = out.logits[:, -1, [zero_id, one_id]].float()
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

                for j, p1 in zip(batch_idxs, probs):
                    p1s[j] = float(p1)
                    preds[j] = int(p1 >= 0.5)

                pos += len(batch_idxs)
                elapsed = time.time() - started
                rate = pos / max(elapsed, 1e-6)
                eta = (len(idxs) - pos) / max(rate, 1e-6) / 60
                print(
                    f"Qwen [{category}] {pos}/{len(idxs)} | batch={bs} | "
                    f"{rate:.2f} product/s | ETA {eta:.1f}m",
                    flush=True,
                )
                del sheets, records, texts, inputs, out, logits, probs
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                bs //= 2
                if bs < 1:
                    raise
                print(f"Qwen OOM -> batch={bs}", flush=True)

    pool.shutdown(wait=True)
    del model, base, processor
    gc.collect()
    torch.cuda.empty_cache()
    return p1s, preds


def comment_for(category, pred):
    if category == BAD:
        if pred == 1:
            return (
                "Карточка соответствует правилам категории БАД: название, описание, "
                "изображения и распознанный текст согласуются с требованиями этой категории."
            )
        return (
            "Карточка не соответствует правилам категории БАД: в названии, описании, "
            "изображениях или распознанном тексте есть признаки несоответствия правилам категории."
        )
    if pred == 1:
        return (
            "Карточка соответствует правилам категории легковоспламеняющихся товаров: "
            "данные карточки и изображений согласуются с установленными критериями проверки."
        )
    return (
        "Карточка не соответствует правилам категории легковоспламеняющихся товаров: "
        "данные карточки или изображений противоречат установленным критериям проверки."
    )


def format_result(category, pred):
    comment = comment_for(category, int(pred))
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
    p1s, preds = run_qwen(df, images_root)
    df["p1"] = p1s
    df["pred"] = preds
    print(f"Qwen stage: {(time.time() - t1) / 60:.2f} min", flush=True)

    df["result"] = [
        format_result(cat, pred)
        for cat, pred in zip(df["category"], df["pred"])
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
