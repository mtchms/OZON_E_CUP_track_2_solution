# Training notebooks финальных адаптеров

В этой папке намеренно оставлены **только два notebook**, относящиеся к двум адаптерам,
которые реально используются в финальных submission.

## 1. OLD BAD

`01_OLD_BAD_ecup2026_phase3_qwen35_qlora.ipynb`

- исторический Phase 3 Qwen3.5-4B QLoRA;
- реальный выполненный BAD training: 7123 train / 346 val;
- после T4 memory patch: LoRA `r=16`, `alpha=32`, `dropout=0.05`;
- именно созданный там BAD позже использовался как `qwen-bad-adapter`;
- финальный подтверждённый SHA256 weights:
  `5a0da02c0bbca13fe0cb7257d85fb407422b705f56d6e449d59acbab1a49984d`.

SHA256 самого notebook в этом репозитории:

`18945730bbf124e74ae60cd2cffb9f2c8e39faec941f23fa05a73a3fdb4a261a`

## 2. NEW FIRE

`02_NEW_FIRE_qwen35_FSDP_FULLTEXT_2xT4_FINAL.ipynb`

- FIRE-only Qwen3.5-4B FSDP-QLoRA;
- 2×T4;
- full description + HQ OCR;
- LoRA `r=16`, `alpha=32`, `dropout=0.05`;
- 5502 unique FIRE rows, 10608 effective balanced rows;
- экспортирует FIRE PEFT adapter;
- финальный подтверждённый SHA256 weights:
  `6bd02b7950e312d69d3e657b1e7bf61d84c1c07a92ae1a7fdb84c0cb00e55d01`.

SHA256 самого notebook в этом репозитории:

`53e7ee114ca5dd5d59126759a8bafbe7babe492a5c71bb981328f2375ed114d1`

## Почему здесь нет остальных notebooks

Ablation, error mining, CLEAN213, TPU/DDP-пробы, NEW BAD и submission-builder notebooks
были исследовательскими промежуточными этапами. Они важны для истории решения, но **не являются
training source для двух адаптеров, реально используемых в финальном сабмите**, поэтому в jury-repo
они не включены.
