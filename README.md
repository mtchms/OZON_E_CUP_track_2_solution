# Ozon E-CUP 2026 — Track 2

## Решение, занявшее 2-е место на Private Leaderboard

Репозиторий содержит финальное решение задачи классификации товаров по двум категориям:

- **БАД**;
- **Легковоспламеняющиеся**.

Организаторы не отображают значение Private score. Из закрытого лидерборда известен только итоговый результат команды — **2-е место**.

Два значения score, приведённые ниже, относятся **только к Public Leaderboard**.

| Вариант inference | Threshold «БАД» | Threshold «Легковоспламеняющиеся» | Public LB |
|---|---:|---:|---:|
| Основной финальный вариант | **0.12** | 0.50 | **0.9042222978144401** |
| Вариант с базовыми threshold | 0.50 | 0.50 | **0.9024064171122994** |

Оба варианта используют **одни и те же веса**. Отличается только threshold для категории «БАД».

![Public LB 0.9042222978144401](assets/public_lb_0.9042222978.png)

---

## Идея решения

В основе используется мультимодальная модель **Qwen3.5-4B** и два независимых LoRA-адаптера — по одному на каждую категорию.

Модель получает:

- название товара;
- описание;
- до пяти изображений;
- OCR-текст, распознанный на изображениях при помощи **PaddleOCR-VL-1.5**.

Схема:

```text
Название ───────────────┐
Описание ───────────────┤
                        │
Изображения ─► contact sheet ──────┐
       │                           │
       └─► PaddleOCR-VL-1.5 ─► OCR text
                                   │
                                   ▼
                           ┌──────────────┐
                           │ Qwen3.5-4B   │
                           │ + LoRA       │
                           └──────┬───────┘
                                  │
                           logits "0" / "1"
                                  │
                                  ▼
                            P(label = 1)
                                  │
                                  ▼
                       threshold по категории
```

Для категорий используются разные адаптеры:

```text
БАД                    -> adapters/bad/
Легковоспламеняющиеся  -> adapters/fire/
```

---

## Структура репозитория

```text
.
├── README.md
├── run.py
├── metadata.json
│
├── adapters/
│   ├── bad/
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   └── fire/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
│
├── notebooks/
│   ├── train_bad_adapter.ipynb
│   └── train_fire_adapter.ipynb
│
├── runtime/
│   ├── bad_t012_fire_t050/
│   │   └── run.py
│   └── bad_t050_fire_t050/
│       └── run.py
│
├── wheels/
│   └── peft-0.20.0-py3-none-any.whl
│
├── scripts/
│   ├── verify_repo.py
│   ├── build_submission.py
│   └── compare_runtimes.py
│
├── docs/
│   ├── ARTIFACTS.md
│   ├── REPRODUCIBILITY.md
│   └── RUNTIME_THRESHOLD_DIFF.patch
│
└── assets/
    ├── public_lb_0.9042222978.png
    └── public_lb_0.9024064171.png
```

`run.py` в корне — точный runtime основного финального варианта с threshold `0.12` для «БАД» и `0.50` для «Легковоспламеняющиеся».

---

# Модели

Базовая модель:

```text
Qwen/Qwen3.5-4B
```

В competition image она доступна по пути:

```text
/shared_models/Qwen/Qwen3.5-4B
```

OCR-модель:

```text
PaddlePaddle/PaddleOCR-VL-1.5
```

В competition image:

```text
/shared_models/PaddlePaddle/PaddleOCR-VL-1.5
```

---

# Обучение адаптеров

В репозитории оставлены только два notebooks, непосредственно относящиеся к обучению адаптеров, использованных в финальном решении.

## Категория «БАД»

Notebook:

```text
notebooks/train_bad_adapter.ipynb
```

Обучение построено как QLoRA fine-tuning Qwen3.5-4B для бинарной классификации.

Ключевые параметры выполненного training run:

```text
LoRA r          = 16
LoRA alpha      = 32
LoRA dropout    = 0.05
learning rate   = 1e-4
epochs          = 1
contact sheet   = 576×576
```

Модель обучалась предсказывать целевую метку напрямую через токены `"0"` и `"1"`.

Notebook содержит сам training pipeline, сохранение PEFT adapter, training summary и validation predictions.

В финальном решении используется следующий бинарный адаптер:

```text
adapters/bad/adapter_model.safetensors
SHA256:
5a0da02c0bbca13fe0cb7257d85fb407422b705f56d6e449d59acbab1a49984d
```

Его config:

```text
adapters/bad/adapter_config.json
SHA256:
291d201a9f049f410c80f8744a9ebda57e4f1ecc10e24904c68ddfec1e8fa391
```

## Категория «Легковоспламеняющиеся»

Notebook:

```text
notebooks/train_fire_adapter.ipynb
```

Используется отдельный OCR-aware QLoRA training pipeline с FSDP на двух NVIDIA T4.

Основные параметры:

```text
base model           = Qwen/Qwen3.5-4B
LoRA r               = 16
LoRA alpha           = 32
LoRA dropout         = 0.05
learning rate        = 1e-4
epochs               = 1
visual side          = 448
contact sheet        = 576×576
unique training rows = 5502
balanced rows        = 10608
text truncation      = none
OCR truncation       = none
```

Training input включает название, полное описание, изображения и OCR-текст.

Notebook поддерживает checkpointing/resume и после FSDP training экспортирует стандартный PEFT adapter.

Финальные веса:

```text
adapters/fire/adapter_model.safetensors
SHA256:
6bd02b7950e312d69d3e657b1e7bf61d84c1c07a92ae1a7fdb84c0cb00e55d01
```

Config:

```text
adapters/fire/adapter_config.json
SHA256:
278b0025d98a0100ef2f5343c69c01aee5bba7029e28dca9642a36ee2330b45b
```

---

# OCR и изображения

Для каждого товара используется до пяти изображений.

Два независимых contact sheet:

```text
OCR:   1024×1024
Qwen:   576×576
```

Основные ограничения inference:

```text
MAX_IMAGES            = 5
MAX_DESCRIPTION_CHARS = 2200
MAX_OCR_CHARS         = 2200
OCR_MAX_NEW_TOKENS    = 160
OCR_BATCH_SIZE        = 32
QWEN_BATCH_SIZE       = 24
```

OCR используется как дополнительный сигнал: текст на упаковке может содержать состав, маркировку, предупреждения и свойства товара, которых нет в обычном описании.

Если OCR завершается ошибкой, submission продолжает работу без OCR и формирует результат для всех строк.

---

# Получение вероятности класса

Вместо генерации свободного текстового ответа для классификации используются logits последней позиции для токенов:

```text
"0"
"1"
```

После softmax вычисляется:

```text
P(label = 1)
```

Для основного финального варианта:

```python
threshold = 0.12 if category == BAD else 0.5
pred = int(p1 >= threshold)
```

Для второго варианта:

```python
pred = int(p1 >= 0.5)
```

Именно изменение threshold для категории «БАД» повысило Public LB:

```text
0.9024064171122994
        ↓
0.9042222978144401
```

Веса и configs между этими двумя отправками не менялись.

---

# Проверка идентичности артефактов

Финальные веса и configs были сверены по двум реально отправленным submission ZIP.

| Артефакт | SHA256 |
|---|---|
| BAD adapter weights | `5a0da02c0bbca13fe0cb7257d85fb407422b705f56d6e449d59acbab1a49984d` |
| BAD adapter config | `291d201a9f049f410c80f8744a9ebda57e4f1ecc10e24904c68ddfec1e8fa391` |
| FIRE adapter weights | `6bd02b7950e312d69d3e657b1e7bf61d84c1c07a92ae1a7fdb84c0cb00e55d01` |
| FIRE adapter config | `278b0025d98a0100ef2f5343c69c01aee5bba7029e28dca9642a36ee2330b45b` |
| Runtime BAD=0.12 / FIRE=0.50 | `a0d89f62d57dedd2cb4e9eaf0eeea606c6b7c58e02336fb9f466d89e7a7b83a5` |
| Runtime BAD=0.50 / FIRE=0.50 | `0fa9229810f90d26d8a1c923c4ed6c4b195e99ba3c7800c94c44f1f16987bd3e` |
| `metadata.json` | `2e84e4db7be7e3f5d6b7b1fae53b0395eb1a6b913704db6dc4f0b39c51095812` |
| PEFT wheel | `0fbba16ffebfad3de96e06f2da6860fd860292324b85b6141909fa1e26ea9233` |

Проверка локального репозитория:

```bash
python scripts/verify_repo.py
```

Успешная проверка заканчивается:

```text
RESULT: ALL OK
```

---

# Сборка submission

Основной финальный вариант:

```bash
python scripts/build_submission.py --variant bad_t012_fire_t050
```

Вариант с threshold `0.50` для обеих категорий:

```bash
python scripts/build_submission.py --variant bad_t050_fire_t050
```

Архивы создаются в:

```text
submission_out/
```

Competition entry point:

```text
python -u run.py
```

Competition image:

```text
odsai/ecup26-quality-baseline:1.0
```

---

# Результат

Лучшая публичная отправка:

```text
Public LB = 0.9042222978144401
```

Итог соревнования:

```text
Private Leaderboard: 2-е место
```

Значение Private score организаторами не отображается.
