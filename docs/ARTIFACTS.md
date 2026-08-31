# Финальные артефакты

## Адаптер «БАД»

```text
adapters/bad/adapter_model.safetensors
SHA256 5a0da02c0bbca13fe0cb7257d85fb407422b705f56d6e449d59acbab1a49984d

adapters/bad/adapter_config.json
SHA256 291d201a9f049f410c80f8744a9ebda57e4f1ecc10e24904c68ddfec1e8fa391
```

## Адаптер «Легковоспламеняющиеся»

```text
adapters/fire/adapter_model.safetensors
SHA256 6bd02b7950e312d69d3e657b1e7bf61d84c1c07a92ae1a7fdb84c0cb00e55d01

adapters/fire/adapter_config.json
SHA256 278b0025d98a0100ef2f5343c69c01aee5bba7029e28dca9642a36ee2330b45b
```

## Runtime

Основной вариант, BAD=0.12 / FIRE=0.50:

```text
runtime/bad_t012_fire_t050/run.py
SHA256 a0d89f62d57dedd2cb4e9eaf0eeea606c6b7c58e02336fb9f466d89e7a7b83a5
```

Вариант BAD=0.50 / FIRE=0.50:

```text
runtime/bad_t050_fire_t050/run.py
SHA256 0fa9229810f90d26d8a1c923c4ed6c4b195e99ba3c7800c94c44f1f16987bd3e
```

## Общие файлы

```text
metadata.json
SHA256 2e84e4db7be7e3f5d6b7b1fae53b0395eb1a6b913704db6dc4f0b39c51095812

wheels/peft-0.20.0-py3-none-any.whl
SHA256 0fbba16ffebfad3de96e06f2da6860fd860292324b85b6141909fa1e26ea9233
```

Значения `0.9042222978144401` и `0.9024064171122994` относятся к **Public Leaderboard**.
Private score не отображается. Итог на Private Leaderboard — **2-е место**.
