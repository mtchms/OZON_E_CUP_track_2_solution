# Notebook inventory

В финальной версии репозитория оставлены только training notebooks, относящиеся к двум
адаптерам, присутствующим в Hybrid A и Private Best.

| Дата | Notebook | Финальный компонент | Статус |
|---|---|---|---|
| 12.08.2026 | `01_OLD_BAD_ecup2026_phase3_qwen35_qlora.ipynb` | OLD BAD | training lineage финального BAD |
| 15.08.2026 | `02_NEW_FIRE_qwen35_FSDP_FULLTEXT_2xT4_FINAL.ipynb` | NEW FIRE | final FIRE-only training |

## OLD BAD provenance

Phase 3 notebook реально обучил `qwen35_4b_BAD_qlora` с `r=16/alpha=32` после memory patch
и сохранил `adapter_model.safetensors`, `adapter_config.json`, `training_summary.json`,
`training_log.csv`, `target_modules.json`, `val_predictions.csv`.

В последующих проверках старый Kaggle input `qwen-bad-adapter` принимался как OLD BAD
только при SHA256 weights:

`5a0da02c0bbca13fe0cb7257d85fb407422b705f56d6e449d59acbab1a49984d`.

## NEW FIRE provenance

FIRE-only notebook обучает и экспортирует `qwen35_4b_FIRE_ocr_fsdp_fulltext`.
Позднее submission builders принимают successful FIRE только при совпадении:

- weights: `6bd02b7950e312d69d3e657b1e7bf61d84c1c07a92ae1a7fdb84c0cb00e55d01`;
- config: `278b0025d98a0100ef2f5343c69c01aee5bba7029e28dca9642a36ee2330b45b`.

Исследовательские notebooks не включены в репозиторий, чтобы jury-версия оставалась компактной.
