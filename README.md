# Ozon E-CUP 2026 - Track 2

## 2-е место на Private Leaderboard

Репозиторий содержит финальное решение задачи классификации товаров по двум категориям:

- БАД
- Легковоспламеняющиеся

Организаторы не отображают значение Private score. Из закрытого лидерборда известен только итоговый результат - 2-е место.

Два score ниже относятся только к Public Leaderboard.

| Вариант inference | Threshold для БАД | Threshold для Легковоспламеняющиеся | Public LB |
|---|---:|---:|---:|
| Основной финальный вариант | **0.12** | 0.50 | **0.9042222978144401** |
| Вариант с threshold 0.50 для обеих категорий | 0.50 | 0.50 | **0.9024064171122994** |

Оба варианта используют одни и те же веса моделей. Отличается только threshold для категории БАД.

![Public LB 0.9042222978144401](assets/public_lb_0.9042222978.png)

## Идея решения

В основе решения используется мультимодальная модель `Qwen/Qwen3.5-4B` и два независимых LoRA-адаптера:

- отдельный адаптер для категории БАД
- отдельный адаптер для категории Легковоспламеняющиеся

Модель использует несколько источников информации о товаре:

- название
- описание
- до 5 изображений
- OCR-текст с изображений

На inference изображения объединяются в contact sheet. Для изображений также выполняется OCR через `PaddlePaddle/PaddleOCR-VL-1.5`. Полученный текст добавляется в prompt Qwen как дополнительный источник информации.

Общая схема:

```text
name + description
        |
        +--------------------------+
        |                          |
        v                          v
   Qwen prompt                product images
                                   |
                         +---------+---------+
                         |                   |
                         v                   v
                  Qwen contact sheet   OCR contact sheet
                                             |
                                             v
                                  PaddleOCR-VL-1.5
                                             |
                                             v
                                          OCR text
                                             |
                         +-------------------+
                         |
                         v
                  Qwen3.5-4B + LoRA
                         |
                         v
                    logits "0"/"1"
                         |
                         v
                    P(label = 1)
                         |
                         v
                category threshold
                         |
                         v
                  final prediction
```

## Гипотезы

## Базовая модель

Используется:

```text
Qwen/Qwen3.5-4B
```

В competition runtime модель доступна по пути:

```text
/shared_models/Qwen/Qwen3.5-4B
```

Для каждой категории загружается свой PEFT/LoRA-адаптер:

```text
adapters/
|- bad/
|  |- adapter_config.json
|  \- adapter_model.safetensors
|
\- fire/
   |- adapter_config.json
   \- adapter_model.safetensors
```

## Обучение адаптера для БАД

Training notebook:

```text
notebooks/train_bad_adapter.ipynb
```

В этом ноутбуке обучались адаптеры для БАД и Легковоспламеняющихся товаров, но в итоге адаптер для Легковоспламеняющихся товаров удалось улучшить, в то время как адаптер для БАД остался неизменным.

Адаптер обучался поверх `Qwen/Qwen3.5-4B` через QLoRA.

Основные параметры выполненного training run:

```text
LoRA r         = 16
LoRA alpha     = 32
LoRA dropout   = 0.05
learning rate  = 1e-4
epochs         = 1
contact sheet  = 576x576
```

Обучение сформулировано как бинарная классификация. Модель обучается выбирать целевой токен `"0"` или `"1"` вместо генерации длинного текстового ответа.

Финальные веса:

```text
adapters/bad/adapter_model.safetensors
SHA256:
5a0da02c0bbca13fe0cb7257d85fb407422b705f56d6e449d59acbab1a49984d
```

Config:

```text
adapters/bad/adapter_config.json
SHA256:
291d201a9f049f410c80f8744a9ebda57e4f1ecc10e24904c68ddfec1e8fa391
```

## Обучение адаптера для Легковоспламеняющиеся

Training notebook:

```text
notebooks/train_fire_adapter.ipynb
```

Для этой категории использовался отдельный OCR-aware training pipeline с `Qwen/Qwen3.5-4B`, QLoRA и FSDP на двух NVIDIA T4.

Основные параметры:

```text
base model           = Qwen/Qwen3.5-4B
training             = FSDP + QLoRA
GPU                  = 2 x NVIDIA T4
LoRA r               = 16
LoRA alpha           = 32
LoRA dropout         = 0.05
learning rate        = 1e-4
epochs               = 1
visual side          = 448
contact sheet        = 576x576
unique training rows = 5502
balanced rows        = 10608
text truncation      = none
OCR truncation       = none
```

В training input входят:

- название
- полное описание
- изображения
- заранее рассчитанный OCR-текст

Training pipeline поддерживает checkpointing и resume. После завершения FSDP training адаптер экспортируется в стандартный PEFT-формат.

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

## OCR всего датасета

Для обучения OCR-aware модели OCR был заранее рассчитан для всего датасета.

Финальный OCR-кэш был получен последовательной обработкой изображений двумя OCR backend'ами.

### 1. PaddleOCR-VL-1.5

Первая часть изображений обрабатывалась моделью:

```text
PaddlePaddle/PaddleOCR-VL-1.5
```

OCR запускался параллельно на двух NVIDIA T4. Результаты сохранялись на уровне отдельных изображений в промежуточные CSV, чтобы обработку можно было продолжать после остановки сессии.

### 2. PP-OCRv5 Mobile

Оставшиеся необработанные изображения были дораспознаны через более быстрый PaddleOCR backend:

```text
detector:
PP-OCRv5_mobile_det

recognizer:
eslav_PP-OCRv5_mobile_rec
```

`eslav_PP-OCRv5_mobile_rec` использовался для распознавания русского и английского текста.

Второй OCR pipeline не начинал обработку заново. Он загружал уже готовые результаты PaddleOCR-VL, пропускал ранее обработанные пары `(product_id, image_idx)` и распознавал только недостающие изображения.

Таким образом, весь датасет был OCR-обработан комбинацией двух backend'ов:

```text
PaddlePaddle/PaddleOCR-VL-1.5
+
PP-OCRv5_mobile_det + eslav_PP-OCRv5_mobile_rec
```

После объединения результатов были сформированы два итоговых файла.

### OCR на уровне изображений

```text
data/ocr/ocr_all_images_high_quality.csv
```

Содержит OCR для отдельных фотографий товара.

Этот файл нужен, если требуется работать с OCR каждого изображения отдельно.

### OCR на уровне товаров

```text
data/ocr/ocr_all_products_high_quality.csv
```

Содержит агрегированный OCR на уровне товара. Текст нескольких фотографий одного товара объединён в одно поле `ocr_text`.

Именно product-level OCR использовался как готовый текстовый признак в OCR-aware training pipeline.

Оба CSV хранятся в Git LFS.

Важно различать precomputed OCR для обучения и OCR на финальном inference:

- для подготовки training data использовался объединённый OCR-кэш от двух backend'ов
- в финальном competition inference OCR считается непосредственно через `PaddlePaddle/PaddleOCR-VL-1.5`

## OCR на inference

Финальный runtime использует:

```text
/shared_models/PaddlePaddle/PaddleOCR-VL-1.5
```

Для каждого товара берётся до 5 изображений.

Основные параметры:

```text
MAX_IMAGES            = 5
OCR contact sheet     = 1024x1024
Qwen contact sheet    = 576x576
MAX_DESCRIPTION_CHARS = 2200
MAX_OCR_CHARS         = 2200
OCR_MAX_NEW_TOKENS    = 160
OCR_BATCH_SIZE        = 32
QWEN_BATCH_SIZE       = 24
```

OCR используется как дополнительный сигнал. Он особенно полезен для текста на упаковке:

- состав
- предупреждения
- маркировка
- свойства товара
- названия веществ
- информация, отсутствующая в текстовом описании карточки

Runtime сделан fault-tolerant. Если OCR не удалось получить, классификация продолжается по доступным тексту и изображениям.

## Получение вероятности класса

Для принятия решения не используется свободная autoregressive generation.

Из последней позиции Qwen берутся logits токенов:

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

Public LB:

```text
0.9042222978144401
```

Для второго сохранённого варианта:

```python
pred = int(p1 >= 0.5)
```

Public LB:

```text
0.9024064171122994
```

Между двумя вариантами не менялись:

- веса адаптера БАД
- config адаптера БАД
- веса адаптера Легковоспламеняющиеся
- config адаптера Легковоспламеняющиеся
- preprocessing
- OCR runtime
- Qwen runtime

Изменился только threshold для категории БАД.

Точный diff двух runtime находится в:

```text
docs/RUNTIME_THRESHOLD_DIFF.patch
```

## Структура репозитория

```text
.
|- README.md
|- run.py
|- metadata.json
|- manifest.json
|
|- adapters/
|  |- bad/
|  |  |- adapter_config.json
|  |  \- adapter_model.safetensors
|  |
|  \- fire/
|     |- adapter_config.json
|     \- adapter_model.safetensors
|
|- data/
|  \- ocr/
|     |- ocr_all_images_high_quality.csv
|     \- ocr_all_products_high_quality.csv
|
|- notebooks/
|  |- train_bad_adapter.ipynb
|  \- train_fire_adapter.ipynb
|
|- runtime/
|  |- bad_t012_fire_t050/
|  |  \- run.py
|  |
|  \- bad_t050_fire_t050/
|     \- run.py
|
|- scripts/
|  |- verify_repo.py
|  |- build_submission.py
|  \- compare_runtimes.py
|
|- docs/
|  |- ARTIFACTS.md
|  |- REPRODUCIBILITY.md
|  \- RUNTIME_THRESHOLD_DIFF.patch
|
|- assets/
|  |- public_lb_0.9042222978.png
|  \- public_lb_0.9024064171.png
|
\- wheels/
   \- peft-0.20.0-py3-none-any.whl
```

## Проверка артефактов

Финальные адаптеры и runtime были сверены по SHA256.

| Артефакт | SHA256 |
|---|---|
| BAD adapter weights | `5a0da02c0bbca13fe0cb7257d85fb407422b705f56d6e449d59acbab1a49984d` |
| BAD adapter config | `291d201a9f049f410c80f8744a9ebda57e4f1ecc10e24904c68ddfec1e8fa391` |
| FIRE adapter weights | `6bd02b7950e312d69d3e657b1e7bf61d84c1c07a92ae1a7fdb84c0cb00e55d01` |
| FIRE adapter config | `278b0025d98a0100ef2f5343c69c01aee5bba7029e28dca9642a36ee2330b45b` |
| Runtime BAD=0.12, FIRE=0.50 | `a0d89f62d57dedd2cb4e9eaf0eeea606c6b7c58e02336fb9f466d89e7a7b83a5` |
| Runtime BAD=0.50, FIRE=0.50 | `0fa9229810f90d26d8a1c923c4ed6c4b195e99ba3c7800c94c44f1f16987bd3e` |
| metadata.json | `2e84e4db7be7e3f5d6b7b1fae53b0395eb1a6b913704db6dc4f0b39c51095812` |
| PEFT wheel | `0fbba16ffebfad3de96e06f2da6860fd860292324b85b6141909fa1e26ea9233` |

Проверить локальный репозиторий:

```bash
python scripts/verify_repo.py
```

Успешный результат:

```text
RESULT: ALL OK
```

## Сборка submission

Основной вариант:

```bash
python scripts/build_submission.py --variant bad_t012_fire_t050
```

Вариант с threshold 0.50 для обеих категорий:

```bash
python scripts/build_submission.py --variant bad_t050_fire_t050
```

Готовые архивы создаются в:

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

## Результат

Лучшая публичная отправка:

```text
Public LB = 0.9042222978144401
```

Итог соревнования:

```text
Private Leaderboard = 2-е место
```

Private score организаторами не отображается.
