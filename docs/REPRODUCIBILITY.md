# Воспроизводимость

## 1. Проверить файлы

```bash
python scripts/verify_repo.py
```

Скрипт проверяет SHA256:

- двух adapter weights;
- двух `adapter_config.json`;
- обоих `run.py`;
- `metadata.json`;
- локального PEFT wheel.

## 2. Собрать основной submission

```bash
python scripts/build_submission.py --variant bad_t012_fire_t050
```

## 3. Собрать вариант с threshold 0.50 / 0.50

```bash
python scripts/build_submission.py --variant bad_t050_fire_t050
```

Оба ZIP используют одни и те же adapters:

```text
adapters/bad/
adapters/fire/
```

и различаются только `run.py`.

## Competition runtime

Ожидаемые shared models:

```text
/shared_models/Qwen/Qwen3.5-4B
/shared_models/PaddlePaddle/PaddleOCR-VL-1.5
```

Entry point:

```text
python -u run.py
```

Основной runtime в корне репозитория (`run.py`) идентичен
`runtime/bad_t012_fire_t050/run.py`.
