# Воспроизводимость

Репозиторий содержит один финальный runtime:

```text
runtime/bad_t012_fire_t050/run.py
```

Корневой `run.py` должен быть побитово идентичен этому файлу.

## Проверка артефактов

Из корня репозитория:

```bash
python scripts/verify_repo.py
```

Скрипт проверяет SHA256 следующих файлов:

- `adapters/bad/adapter_model.safetensors`
- `adapters/bad/adapter_config.json`
- `adapters/fire/adapter_model.safetensors`
- `adapters/fire/adapter_config.json`
- `run.py`
- `runtime/bad_t012_fire_t050/run.py`
- `metadata.json`
- `wheels/peft-0.20.0-py3-none-any.whl`

Также проверяется, что корневой `run.py` побитово идентичен
`runtime/bad_t012_fire_t050/run.py`.

При успешной проверке:

```text
RESULT: ALL OK
```

## Сборка submission

```bash
python scripts/build_submission.py --variant bad_t012_fire_t050
```

В submission используются:

```text
run.py
metadata.json

adapters/
  bad/
    adapter_config.json
    adapter_model.safetensors

  fire/
    adapter_config.json
    adapter_model.safetensors

wheels/
  peft-0.20.0-py3-none-any.whl
```

## Competition runtime

Ожидаемые shared models:

```text
/shared_models/Qwen/Qwen3.5-4B
/shared_models/PaddlePaddle/PaddleOCR-VL-1.5
```

Entry point:

```bash
python -u run.py
```

Финальные thresholds:

```text
БАД: 0.12
Легковоспламеняющиеся: 0.50
```

## Финальная проверка перед отправкой

```bash
python scripts/verify_repo.py
python scripts/build_submission.py --variant bad_t012_fire_t050
```

После этого необходимо убедиться, что итоговый ZIP содержит `run.py`,
`metadata.json`, оба LoRA adapter и локальный PEFT wheel.
