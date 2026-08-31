# Артефакты

В репозитории сохранены только артефакты финального решения.

## Runtime

Основной competition entrypoint:

```text
run.py
```

Зафиксированная копия финального runtime:

```text
runtime/bad_t012_fire_t050/run.py
```

Оба файла должны быть побитово идентичны.

SHA256:

```text
a0d89f62d57dedd2cb4e9eaf0eeea606c6b7c58e02336fb9f466d89e7a7b83a5
```

Runtime использует thresholds:

```text
БАД: 0.12
Легковоспламеняющиеся: 0.50
```

## BAD adapter

```text
adapters/bad/adapter_model.safetensors
```

SHA256:

```text
5a0da02c0bbca13fe0cb7257d85fb407422b705f56d6e449d59acbab1a49984d
```

Конфигурация:

```text
adapters/bad/adapter_config.json
```

SHA256:

```text
291d201a9f049f410c80f8744a9ebda57e4f1ecc10e24904c68ddfec1e8fa391
```

## FIRE adapter

```text
adapters/fire/adapter_model.safetensors
```

SHA256:

```text
6bd02b7950e312d69d3e657b1e7bf61d84c1c07a92ae1a7fdb84c0cb00e55d01
```

Конфигурация:

```text
adapters/fire/adapter_config.json
```

SHA256:

```text
278b0025d98a0100ef2f5343c69c01aee5bba7029e28dca9642a36ee2330b45b
```

## Metadata

```text
metadata.json
```

SHA256:

```text
2e84e4db7be7e3f5d6b7b1fae53b0395eb1a6b913704db6dc4f0b39c51095812
```

## PEFT wheel

Для воспроизводимого offline runtime используется локальный wheel:

```text
wheels/peft-0.20.0-py3-none-any.whl
```

SHA256:

```text
0fbba16ffebfad3de96e06f2da6860fd860292324b85b6141909fa1e26ea9233
```

## Shared models competition environment

Base model:

```text
/shared_models/Qwen/Qwen3.5-4B
```

OCR model, используемая непосредственно во время inference:

```text
/shared_models/PaddlePaddle/PaddleOCR-VL-1.5
```

Эти shared models не хранятся в репозитории.

## Training notebooks

Код обучения финальных LoRA adapter находится в:

```text
notebooks/train_bad_adapter.ipynb
notebooks/train_fire_adapter.ipynb
```

## Предрасчитанный OCR

OCR-кэш для training pipeline находится в:

```text
data/ocr/ocr_all_images_high_quality.csv
data/ocr/ocr_all_products_high_quality.csv
```

`ocr_all_images_high_quality.csv` содержит OCR на уровне отдельных изображений.

`ocr_all_products_high_quality.csv` содержит объединенный OCR на уровне товара и
используется OCR-aware training pipeline.

## Проверка

Все зафиксированные SHA256 можно проверить одной командой:

```bash
python scripts/verify_repo.py
```

Ожидаемый результат:

```text
RESULT: ALL OK
```
