# OCR and multimodal inference pipeline

## Why OCR mattered

Many products contain decisive evidence only on packaging: `БАД`, `биологически активная добавка`, `dietary supplement`, warnings, fuel/gas markings, or wording that is absent from the marketplace description. We therefore treated OCR as an **additional modality**, not as a replacement for title/description/image understanding.

The project passed through several OCR stages:

1. no-OCR text+image baselines;
2. legacy PPOCRv5/mobile experiments;
3. explicit BASE-vs-OCR ablation;
4. isolated official PaddleOCR-VL-1.5 pipeline;
5. OCR-aware adapter training experiments;
6. high-quality product-level OCR for the CLEAN213 branch;
7. final robust runtime where OCR failure does not prevent producing a submission.

## Final runtime

- OCR model: `/shared_models/PaddlePaddle/PaddleOCR-VL-1.5`
- classifier/base VLM: `/shared_models/Qwen/Qwen3.5-4B`
- up to **5** product images;
- OCR contact sheet: **1024×1024**;
- Qwen contact sheet: **576×576**;
- OCR max/min pixels: `1003520 / 112896`;
- OCR generation cap: **160** new tokens;
- OCR batch size: **32**;
- Qwen batch size: **24**;
- title/description and OCR are combined with category rules in the classification prompt;
- description and OCR runtime caps: **2200 chars each** in the final old runtime lineage.

OCR runs before adapter-specific Qwen inference for every product with images. The recognized text is inserted into the prompt with an explicit warning that OCR may contain mistakes.

## Runtime resilience

The submission was designed for a constrained evaluator:

- offline Hugging Face/Transformers mode;
- local shared models;
- FlashAttention-2 when available, SDPA fallback otherwise;
- CUDA OOM batch-halving;
- OCR soft time budget = **55%** of the total runtime budget;
- if OCR hits the soft limit, remaining products continue with empty OCR;
- if OCR fails completely, Qwen still processes every row without OCR;
- output completeness and format are validated before writing the CSV.

This failure-safe behavior was important: OCR could improve difficult packaging-heavy cases without turning OCR into a single point of failure.

## Training OCR versus inference OCR

A major lesson was that these are separate design choices.

- The historical **OLD BAD** that survived into the final solution was trained in the earlier lineage, while the final runtime still supplied PaddleOCR-VL text at inference.
- The successful **NEW FIRE** came from the OCR-aware FIRE training lineage.
- The **CLEAN213 HQ-OCR BAD** branch deliberately trained with full product-level HQ OCR and also used OCR at inference, but it did not ultimately replace OLD BAD in the private-best solution.

So the final result came from *controlled specialization*, not from forcing both categories to use the same training recipe.
