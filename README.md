# Криптонит: Тембр — Speaker Retrieval

## Что за задача

Соревнование по **распознаванию диктора**. Для каждой из 134 697 тестовых аудиозаписей нужно найти 10 наиболее похожих записей из того же тестового набора и сдать результат в виде `submission.csv`.

Задача называется **speaker retrieval** (поиск по голосу):
- у каждой записи нет метки в тестовом наборе — метки есть только в обучающей выборке;
- нужно сравнить записи между собой по **speaker embedding** (векторному представлению голоса);
- качество измеряется метрикой **Precision@K** (доля соседей, принадлежащих тому же диктору).

---

## Данные

### Обучающая выборка (`data/train/`)

| Параметр | Значение |
|---|---|
| Файлов | 673 277 |
| Дикторов | 11 053 |
| Файлов на диктора | min=1, median=64, max=100 |
| Формат | FLAC, 16 kHz |
| Разметка | `train.csv`: `speaker_id`, `filepath` |
| Размер на диске | ~101 GB |

Структура: `data/train/{speaker_id}/{NNNNN}.flac`

### Тестовая выборка (`data/test_public/`)

| Параметр | Значение |
|---|---|
| Файлов | 134 697 |
| Меток дикторов | нет |
| Формат | FLAC, 16 kHz |
| Шаблон | `test_public.csv`: колонка `filepath` |
| Размер на диске | ~18 GB |

Структура: `data/test_public/{NNNNNN}.flac` (имя файла = его индекс в CSV)

### Источник

Данные получены от организаторов соревнования в виде архивов:
- `train_part_1..10.tar.gz` — обучающие данные
- `test_public.tar.gz` — тестовые данные
- `baseline.onnx` — предобученная модель от организаторов

---

## Что такое baseline и что мы обучаем

### Baseline (от организаторов)

`extracted_data/Для участников/baseline.onnx` — это **готовая предобученная модель**, которую организаторы соревнования выдали участникам как стартовую точку. Это ECAPA-TDNN весом 31 MB, экспортированный в формат ONNX.

- **Вход**: сырой аудиосигнал float32, форма `[batch, time]`, 16 kHz
- **Выход**: speaker embedding, форма `[batch, 192]`
- Модель обработана через `CUDAExecutionProvider` (ONNX Runtime)

Мы использовали её чтобы:
1. Убедиться что пайплайн работает корректно (embedding → FAISS → submission)
2. Зафиксировать **baseline метрику** на нашем локальном val-сплите

**Baseline метрики (val-сплит, 500 дикторов, 30 553 файла):**

| P@1 | P@5 | P@10 |
|---|---|---|
| 0.9735 | 0.9348 | **0.8941** |

### Наша модель

Файл: `src/models/ecapa_tdnn.py`

#### v1 — обучение с нуля, синтетические аугментации

Конфиг: `configs/train_ecapa.yaml`
Чекпоинт: `weights/ecapa_tdnn_aamsoft_best.pt` (эпоха 27, val P@10=0.9458)

| Эпоха | Loss | P@1 | P@5 | P@10 |
|---|---|---|---|---|
| Baseline ONNX | — | 0.9735 | 0.9348 | 0.8941 |
| 1 | 12.99 | 0.8964 | 0.8109 | 0.7417 |
| 5 | 3.62 | 0.9797 | 0.9497 | 0.9158 |
| 27 | — | — | — | **0.9458** (best) |

Public P@10: **0.2699** — провал из-за отсутствия реальных шумов/RIR.

#### v2 — дообучение с реальными MUSAN + RIR, SubcenterArcFace

Конфиг: `configs/train_ecapa_v2.yaml`
Старт: чекпоинт v1 (эпоха 27), reset best → 30 дополнительных эпох (28–57)
Loss: SubcenterArcFace (K=3, margin=0.2, scale=30)
Аугментации: MUSAN noise (930 файлов, p=0.5) + RIR (60k файлов, p=0.6)
Чекпоинт: `weights/ecapa_tdnn_v2_realaugment_best.pt` (эп.57, val P@10=0.9465)
Public P@10: **0.3717**

| Эпоха | Loss | P@1 | P@5 | P@10 |
|---|---|---|---|---|
| 28 | 10.28 | 0.9812 | 0.9596 | 0.9363 |
| 50 | 3.08 | 0.9839 | 0.9663 | 0.9460 |
| 57 | 3.02 | 0.9846 | 0.9668 | **0.9465** (best) |

**Анализ**: val P@10=0.9465 (чистые данные), public P@10=0.3717 — разрыв сохраняется. MUSAN+RIR улучшили результат vs v1 (+0.102), но CAM++ zero-shot (0.4560) всё равно лучше — он обучен на более разнообразных данных (VoxCeleb1+2).

---

## Архитектура модели

### ECAPA-TDNN (`src/models/ecapa_tdnn.py`)

```
Вход: waveform (B, T) float32
  ↓
LogMelFrontend: MelSpectrogram(n_mels=80, hop=10ms, win=25ms) + log
  ↓ (B, 80, T')
Conv1d + BN + ReLU
  ↓ (B, 512, T')
SERes2Block(dilation=2) → SERes2Block(dilation=3) → SERes2Block(dilation=4)
  ↓ три ветки конкатенируются → (B, 1536, T')
Conv1d агрегация
  ↓
AttentiveStatsPool → (B, 3072)   # mean + std с attention-весами
  ↓
BatchNorm → Linear(3072, 192) → BatchNorm
  ↓
L2-normalize
  ↓
Выход: embedding (B, 192) float32, ||e|| = 1
```

**Параметры: 6.2M**

Ключевые блоки:
- **SERes2Block**: Res2-свёртки с dilated convolution + Squeeze-Excitation (канальное внимание)
- **AttentiveStatsPool**: взвешенные среднее и стандартное отклонение по времени (внимание к наиболее информативным фреймам)
- **LogMelFrontend**: встроен в модель (не зависит от внешнего препроцессинга)

### Loss: AAM-Softmax (`src/models/loss.py`)

Additive Angular Margin Softmax (ArcFace) — стандартный loss для speaker verification:
```
cos(θ + m) для правильного класса, где m = 0.2 (≈11.5°)
scale = 30
n_classes = 10 553 (дикторов в train-части)
```

Идея: принудительно создавать угловой зазор между кластерами разных дикторов в embedding-пространстве.

---

## Аугментации (`src/data/augmentation.py`)

Применяются на лету во время обучения для имитации реальных условий записи:

| Аугментация | v1 | v2 | Описание |
|---|---|---|---|
| Белый шум | 30% | 20% | Гауссов шум, SNR 0–15 dB |
| Розовый шум | 20% | 10% | 1/f шум, SNR 0–15 dB |
| Шум из файлов (MUSAN) | ❌ | **50%** | 930 реальных noise-файлов |
| Реверберация (RIR) | ❌ | **60%** | 60k impulse responses (OpenSLR) |
| Телефонный канал | 30% | 30% | Bandpass 300–3400 Hz + ресемплинг до 8 kHz и обратно |
| Кодек (MP3) | 20% | 20% | MP3 с низким битрейтом (8–32 kbps) |
| Speed perturbation | 30% | 30% | Скорость ×0.9 / ×1.0 / ×1.1 |
| Volume jitter | 50% | 50% | Случайное усиление ±6 dB |

Порядок применения: volume → speed → noise (один тип) → RIR → codec/telephone.

В v1 реальные шумы и RIR были недоступны → использовались только синтетические аугментации → провал на тесте.

---

## Retrieval-пайплайн

После извлечения эмбеддингов поиск соседей работает так:

```
embeddings (N, 192)
    ↓ L2-нормализация
normalized (N, 192), ||e_i|| = 1
    ↓ FAISS IndexFlatIP (точный поиск по inner product = cosine similarity)
    ↓ на GPU: RTX 3090, поиск по 134k векторам за < 1 секунды
    ↓ запрос k+1=11 соседей, удаление self-index i
top-10 neighbours (N, 10)
    ↓
submission.csv
```

FAISS `IndexFlatIP` + L2-нормализованные вектора = точный поиск по косинусному сходству.

---

## Как тестируем и валидируем

### Локальная оценка (offline validation)

**Сплит**: из 11 053 дикторов train-набора берём 500 случайных (seed=42) — это **val-set**. Остальные 10 553 идут на обучение.

**Процедура**:
1. Извлекаем эмбеддинги всех val-файлов (~30 553 файла)
2. Строим FAISS-индекс на val-эмбеддингах
3. Для каждого val-файла ищем 10 ближайших соседей (исключая себя)
4. Считаем `Precision@K` = доля соседей с тем же `speaker_id`
5. Усредняем по всем файлам

**Запуск**:
```bash
bash scripts/run.sh scripts/run_eval.py \
    --data-root data/ \
    --train-csv "extracted_data/Для участников/train.csv" \
    --model "extracted_data/Для участников/baseline.onnx" \
    --n-val-speakers 500 \
    --experiment-name baseline_onnx
```

### Что важно в этом сплите

- Сплит **по дикторам**, не по файлам — один диктор не может одновременно быть в train и val
- Один и тот же seed=42 используется для всех экспериментов → сравнения честные
- Метрика на val-сплите **не гарантирует** результат на private leaderboard, но является надёжным ориентиром

### Итоговая проверка submission

После каждого инференса автоматически запускается `validate_submission()`:
- ровно 134 697 строк
- порядок `filepath` совпадает с шаблоном
- ровно 10 соседей на строку
- нет NaN, нет дублей, нет self-index
- все индексы в диапазоне [0, 134696]

---

## Обучение

### Параметры

```yaml
batch_size: 256
lr: 1e-3
weight_decay: 2e-5
epochs: 30
chunk_s: 3.0        # каждый файл кропается до 3 секунд случайно
warmup_epochs: 2    # линейный warmup LR
scheduler: cosine   # cosine annealing после warmup
grad_clip: 5.0
num_workers: 8
```

### BF16 (Brain Float 16)

RTX 3090 — архитектура **Ampere**, поддерживает аппаратные BF16 tensor cores.
В тренере включён `torch.autocast(dtype=torch.bfloat16)`:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    embeddings = encoder(waveforms)
    loss = loss_fn(embeddings, labels)
```

Эффект:
- ~1.5–2x ускорение обучения
- Меньше VRAM → можно увеличить batch_size
- Точность не теряется (BF16 имеет тот же диапазон экспоненты что и FP32)

> **Flash Attention 2** — не применимо к ECAPA-TDNN (нет трансформерного self-attention).
> Станет актуальным при переходе на WavLM.

Управляется флагом в конфиге: `training.bf16: true`

### Скорость

- ~4 it/s (FP32) → ожидается ~6–7 it/s с BF16
- GPU: RTX 3090, ~15 GB VRAM
- 30 эпох ≈ 5.5 часов (FP32) → ~3.5 часа (BF16)

### Запуск обучения

```bash
bash scripts/run.sh scripts/run_train.py --config configs/train_ecapa.yaml
```

Лог: `results/train_ecapa.log`
Чекпоинт лучшей эпохи (по P@10): `weights/ecapa_tdnn_aamsoft_best.pt`

---

## Инференс

### С baseline ONNX (организаторы)

```bash
bash scripts/run.sh scripts/run_baseline.py \
    --data-root data/ \
    --test-csv  "extracted_data/Для участников/test_public.csv" \
    --model     "extracted_data/Для участников/baseline.onnx" \
    --output    submission_baseline.csv \
    --batch-size 128 \
    --cache-dir embeddings/
```

### С нашей обученной моделью

```bash
bash scripts/run.sh scripts/run_inference_torch.py \
    --checkpoint weights/ecapa_tdnn_aamsoft_best.pt \
    --config     configs/train_ecapa.yaml \
    --data-root  data/ \
    --test-csv   "extracted_data/Для участников/test_public.csv" \
    --output     submission.csv \
    --batch-size 128 \
    --cache-dir  embeddings/
```

Время инференса: ~4–5 минут на 134 697 файлов (GPU).

---

## Структура репозитория

```
ASR/
├── README.md
├── pyproject.toml                  # зависимости
├── .gitignore
│
├── configs/
│   ├── train_ecapa.yaml            # конфиг обучения ECAPA-TDNN
│   └── baseline.yaml               # конфиг для baseline ONNX (TODO)
│
├── src/
│   ├── data/
│   │   ├── dataset.py              # FlacDataset, collate_pad
│   │   ├── train_dataset.py        # SpeakerTrainDataset (с аугментацией)
│   │   └── augmentation.py         # AugmentationPipeline
│   ├── models/
│   │   ├── ecapa_tdnn.py           # ECAPA-TDNN (наша модель)
│   │   ├── onnx_wrapper.py         # обёртка над baseline.onnx
│   │   └── loss.py                 # AAMSoftmax, SubcenterArcFace
│   ├── inference/
│   │   └── extract_embeddings.py   # батчинг + кэш + prefetch
│   ├── retrieval/
│   │   └── faiss_search.py         # build_index, find_neighbors
│   ├── training/
│   │   └── trainer.py              # training loop, evaluation, checkpointing
│   └── utils/
│       ├── audio.py                # load_audio, normalize, VAD
│       ├── metrics.py              # precision_at_k
│       ├── submission.py           # save_submission, validate_submission
│       └── experiment_logger.py    # JSON-L лог экспериментов
│
├── scripts/
│   ├── run.sh                      # обёртка: LD_LIBRARY_PATH + venv + PYTHONPATH
│   ├── run_baseline.py             # инференс через ONNX
│   ├── run_inference_torch.py      # инференс через PyTorch чекпоинт
│   ├── run_train.py                # запуск обучения (--resume, --reset-best)
│   ├── run_eval.py                 # локальная оценка P@K
│   ├── run_pretrained.py           # zero-shot: SpeechBrain / WavLM (attention_mask fix)
│   ├── run_campplus.py             # zero-shot: CAM++ (ModelScope, GPU batching)
│   └── extract_archives.sh         # распаковка tar.gz архивов
│
├── data/                           # (gitignored) распакованные аудиофайлы
│   ├── train/{speaker_id}/*.flac
│   └── test_public/*.flac
│
├── embeddings/                     # (gitignored) кэш .npy эмбеддингов
├── weights/                        # (gitignored) чекпоинты моделей
├── results/
│   ├── experiments.jsonl           # лог всех экспериментов
│   └── train_ecapa.log             # лог текущего обучения
│
└── extracted_data/                 # оригинальные файлы организаторов
    └── Для участников/
        ├── baseline.onnx
        ├── train.csv
        ├── test_public.csv
        └── *.tar.gz
```

---

## Установка окружения

```bash
# Python 3.12, uv
uv venv
source .venv/bin/activate
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install onnxruntime-gpu onnx soundfile librosa pandas numpy scipy \
               scikit-learn tqdm omegaconf faiss-gpu-cu12
```

> **Важно**: `scripts/run.sh` автоматически добавляет CUDA-библиотеки из venv в `LD_LIBRARY_PATH`, что необходимо для работы `onnxruntime-gpu`. Всегда запускайте скрипты через `bash scripts/run.sh <script> <args>`.

---

## Запрещено (правила соревнования)

- Использовать датасет **VoxBlink2** или производные от него модели
- Решения, которые нельзя воспроизвести по инструкции

---

## Результаты сабмитов

### Сабмит 1 — ECAPA-TDNN, обучен с нуля, синтетические аугментации

| | Val P@10 (локально) | Public leaderboard P@10 |
|---|---|---|
| Baseline ONNX (организаторы) | 0.8941 | — |
| ECAPA-TDNN эпоха 27/30 | 0.9458 | **0.2699** |

**Место: 14/29. Команда: AI Amigos.**

#### Анализ провала

Разрыв val→test: **0.9458 → 0.2699** — катастрофический.

**Причина**: val-сплит составлен из train-данных (чистый звук), тестовый набор содержит сильно искажённые записи (шум, реверберация, far-field, кодеки). Наша модель обучена на синтетических аугментациях без реальных шумовых файлов:
- `p_rir=0.5` — реверберация **не работала** (RIR-файлов не было)
- `p_file_noise=0.4` — реальный шум **не работал** (noise-файлов не было)
- Использовались только: белый/розовый шум, телефония, speed perturbation

Лидер (0.7140) почти наверняка использует реальные шумы и RIR.

---

### Сабмит 2 — SpeechBrain ECAPA-TDNN, zero-shot (VoxCeleb1+2)

Предобученная модель `speechbrain/spkrec-ecapa-voxceleb` без какого-либо дообучения.

| | Public leaderboard P@10 |
|---|---|
| SpeechBrain ECAPA (zero-shot) | **0.3858** |

**Вывод**: zero-shot модель на VoxCeleb1+2 (~7000 дикторов) превзошла нашу модель +0.116. Проблема — в аугментациях, а не в архитектуре.

#### Лидерборд (public, на момент сабмита)
| # | Команда | P@10 |
|---|---|---|
| 1 | Авантюрист | 0.7140 |
| 2 | PC squeaker | 0.6365 |
| 3 | KIWI-7200 | 0.6100 |
| 4 | lab260 | 0.5978 |
| 5 | Команда | 0.5881 |
| ... | ... | ... |
| **14** | **AI Amigos** | **0.3858** |

---

### Сабмит 5 — Ансамбль CAM++ + ECAPA v2 (grid search по alpha)

Конкатенация L2-нормализованных эмбеддингов: `[α × emb_campplus | (1-α) × emb_ecapa_v2]` → 704-dim → L2-normalize → FAISS.

| alpha (CAM++) | alpha (ECAPA v2) | Public P@10 |
|---|---|---|
| 0.8 | 0.2 | 0.4658 |
| **0.7** | **0.3** | **0.4776** ← лучший |

Скрипт: `scripts/run_ensemble.py --grid-search`

---

### Сабмит 3 — CAM++ zero-shot (Alibaba DAMO, VoxCeleb)

`damo/speech_campplus_sv_en_voxceleb_16k` через ModelScope. GPU batching (256 файлов/батч, ~100 файлов/сек).

| | Public leaderboard P@10 |
|---|---|
| CAM++ zero-shot | **0.4560** |

**+0.070** к SpeechBrain ECAPA. Место: **17/38**.

---

### Сабмит 4 — WavLM-Base-Plus-SV, zero-shot

| | Public leaderboard P@10 |
|---|---|
| WavLM-Base-Plus-SV v1 (двойная нормализация) | 0.1167 ❌ |
| WavLM-Base-Plus-SV v2 (исправлено) | 0.1149 ❌ |

**Причина провала**: `collate_pad` паддит батч нулями. WavLM без `attention_mask` обрабатывает нули как реальный сигнал → испорченные эмбеддинги. Исправлено: `WavLMEmbedder.embed_batch` теперь обрезает trailing zeros и передаёт переменной длины массивы в feature extractor (который сам создаёт корректный attention_mask).

---

## Перспективы улучшения

### Итоговая таблица результатов

| Модель | Val P@10 | Public P@10 | Место |
|---|---|---|---|
| Baseline ONNX | 0.8941 | — | — |
| ECAPA v1 (синт. аугм.) | 0.9458 | 0.2699 | 14/29 |
| WavLM zero-shot | — | 0.1149 | — |
| SpeechBrain ECAPA zero-shot | — | 0.3858 | — |
| ECAPA v2 (MUSAN+RIR) | 0.9465 | 0.3717 | — |
| **CAM++ zero-shot** | — | **0.4560** | **17/38** |

### Анализ ситуации

Val P@10 не коррелирует с public P@10 — val набор чистый, test сильно зашумлён. Главный конкурент — CAM++, обученный на VoxCeleb с бо́льшим разнообразием данных. Наша ECAPA-TDNN уступает, несмотря на MUSAN+RIR аугментации.

### Ближайшие шаги (по приоритету)

| # | Шаг | Ожидаемый эффект | Сложность |
|---|---|---|---|
| 1 | **Ансамбль CAM++ + ECAPA v2** (усреднение эмбеддингов) | +0.02–0.05 | Низкая (эмбеддинги уже есть) |
| 2 | **Fine-tune CAM++** на наших данных с MUSAN+RIR | +0.05–0.15 | ✅ Запущено (20 эп., ~14ч) |
| 3 | **TTA** (test-time augmentation: несколько аугм. версий → avg emb) | +0.02–0.06 | Средняя |
| 4 | **Обучение на train+val** (все 11 053 диктора) | +0.01–0.03 | Низкая |
| 5 | **WavLM-Base-Plus-SV** с исправленным attention_mask | +0.05–0.10? | Низкая |
| 6 | **Больший ECAPA** или ERes2Net | +0.03–0.08 | Высокая |

### Архитектурные эксперименты

#### ❌ Не подходят для speaker retrieval
| Модель | Причина |
|---|---|
| Whisper v3 / Turbo | ASR-модель: учит **что** сказано, не **кто** говорит |
| Wav2Vec 2.0 / XLSR-53 | Content-ориентированы; можно как feature extractor, но слабее специализированных |

#### ✅ Подходят (по убыванию приоритета)
| Модель | Почему | Сложность |
|---|---|---|
| **WavLM-Large** (Microsoft) | Предобучен на зашумлённых данных, SOTA на VoxCeleb, робастен к шуму по дизайну | Средняя |
| **WavLM-Base-Plus-SV** | Уже дообучен под speaker verification, грузится через HuggingFace двумя строками | Низкая |
| **CAM++** (Alibaba/ModelScope) | Специально для speaker verification, быстрый, конкурирует с ECAPA при меньшем размере | Низкая |
| **ERes2Net** | Улучшенный Res2Net для speaker verification, сильный результат на VoxCeleb2 | Средняя |
| **ReDimNet** (2024) | Новая архитектура, SOTA на ряде бенчмарков | Средняя |

### Прогноз по public P@10

| Конфигурация | Ожидаемый результат |
|---|---|
| Текущий результат (ECAPA + синт. аугм.) | 0.2699 |
| + реальные MUSAN/RIR | 0.45–0.60 |
| + SubcenterArcFace | 0.50–0.62 |
| WavLM-Base-Plus-SV + реальные аугм. | 0.60–0.70 |
| WavLM-Large + дообучение + реальные аугм. | **0.65–0.75+** |

---

## Текущий статус

- [x] Данные распакованы (101 GB train + 18 GB test)
- [x] Baseline ONNX-инференс работает
- [x] Локальная оценка P@K настроена
- [x] ECAPA-TDNN обучен (30 эпох, val P@10=0.9458)
- [x] Сабмит 1: ECAPA-TDNN с нуля + синт. аугм. → **0.2699**
- [x] Сабмит 2: SpeechBrain ECAPA zero-shot → **0.3858**
- [x] Сабмит 3: WavLM-Base-Plus-SV zero-shot (с багом двойной норм.) → **0.1167** ❌
- [x] Сабмит 4: WavLM-Base-Plus-SV zero-shot (исправлено) → **0.1149** ❌

> WavLM стабильно плохой результат (~0.115). Причина: наш `collate_pad` паддит батч нулями до максимальной длины, WavLM обрабатывает нули как реальный сигнал. SpeechBrain к этому устойчив, WavLM — нет. Требует отдельного пайплайна (по одному файлу или с attention_mask). Отложено.
- [x] Скачаны MUSAN (930 noise файлов) + RIR (60k файлов)
- [x] BF16 autocast добавлен в тренер
- [x] Флаги `--resume` / `--reset-best` добавлены в run_train.py
- [x] WavLM attention_mask fix: trailing zeros обрезаются перед feature extractor
- [x] CAM++ zero-shot: установлен ModelScope, переписан под GPU batching (~100 файлов/сек)
- [x] ECAPA-TDNN v2: 30 эпох (28→57), SubcenterArcFace, MUSAN+RIR — **запущено**
  - Эпоха 28 завершена: P@10=0.9363, чекпоинт сохранён
- [x] CAM++ zero-shot → **0.4560** (место 17/38)
- [x] ECAPA v2 (MUSAN+RIR, SubcenterArcFace, эп.28–57) → **0.3717**
- [x] Ансамбль CAM++ α=0.7 + ECAPA v2 α=0.3 → **0.4776** ← лучший результат
- [x] Ансамбль CAM++ α=0.8 + ECAPA v2 α=0.2 → 0.4658
- [x] CAM++ fine-tune запущен (PID 293598, 10 эпох, GPU frontend, ~2.8ч)
- [ ] Инференс и сабмит fine-tuned CAM++
- [ ] Ансамбль fine-tuned CAM++ + ECAPA v2 (alpha≈0.7)
- [ ] TTA (test-time augmentation)
- [ ] Попробовать WavLM с исправленным attention_mask
- [ ] EDA-ноутбук
- [ ] Финальный submission
- [ ] Отчёт и презентация
