# Криптонит: Тембр — Speaker Retrieval

**Команда:** AI Amigos | **Результат:** Public P@10 = 0.7439 (6-е место)

Решение задачи поиска похожих записей по голосу. Для каждой из 134 697 тестовых FLAC-записей находим 10 ближайших соседей по speaker embedding.

> Подробное описание подходов, экспериментов и выводов — в [REPORT.md](REPORT.md).

---

## Быстрый старт (Docker)

> Инференс работает **полностью офлайн** — все модели и зависимости кэшируются в образе на этапе сборки.

### Шаг 1 — Подготовка (требуется интернет)

```bash
# 1. Собрать образ (кэширует ModelScope модели внутри)
docker build -t kryptonit-tembr .

# 2. Скачать fine-tuned веса
mkdir -p weights
python3 -c "
from huggingface_hub import hf_hub_download
for f in ['campplus_finetune_stage3_best.pt', 'eres2net_finetune_stage3_best.pt']:
    hf_hub_download('s0ft44/kryptonit-tembr-weights', f, local_dir='weights/')
print('Weights ready.')
"
```

Репозиторий с весами: https://huggingface.co/s0ft44/kryptonit-tembr-weights

### Шаг 2 — Инференс (работает без интернета)

```bash
docker run --gpus all \
    -v /path/to/data:/app/data \
    -v $(pwd)/weights:/app/weights \
    -v $(pwd)/output:/app/output \
    kryptonit-tembr \
    --data-root /app/data \
    --test-csv /app/data/test_public.csv \
    --output /app/output/submission.csv
```

На выходе: `output/submission.csv` с 10 соседями для каждой записи.

---

## Запуск без Docker

```bash
# Установка окружения
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Скачать веса (с интернетом)
python3 -c "
from huggingface_hub import hf_hub_download
for f in ['campplus_finetune_stage3_best.pt', 'eres2net_finetune_stage3_best.pt']:
    hf_hub_download('s0ft44/kryptonit-tembr-weights', f, local_dir='weights/')
"

# Инференс (работает без интернета)
python infer.py \
    --data-root data/ \
    --test-csv extracted_data/test_public.csv \
    --output submission.csv
```

---

## Ожидаемая структура данных

```
data/
└── test_public/
    ├── 000000.flac
    ├── 000001.flac
    └── ...
extracted_data/
└── test_public.csv     # колонка filepath: test_public/NNNNNN.flac
weights/                # скачать на Шаге 1
output/                 # сюда сохраняется submission.csv
```

---

## Время инференса

| Этап | Время |
|---|---|
| CAM++ TTA×10, 134k файлов | ~60 мин |
| ERes2Net TTA×10, 134k файлов | ~60 мин |
| K-reciprocal re-ranking | ~12 мин |
| **Итого** | **~2.5 часа** |

---

## Параметры инференса

| Параметр | Значение | Описание |
|---|---|---|
| `--alpha` | 0.55 | Вес CAM++ в ансамбле (1−α для ERes2Net) |
| `--k1` | 70 | K-reciprocal neighbourhood size |
| `--k2` | 6 | Query expansion size |
| `--lambda-val` | 0.1 | Вес косинусной дистанции в re-ranking |
| `--n-crops` | 10 | Число TTA-кропов на запись |
| `--batch-size` | 64 | Батч инференса |

---

## Запрещено (правила соревнования)

- Датасет **VoxBlink2** и производные от него модели
