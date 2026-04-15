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

### Как аугментации накладываются

Аугментации применяются **на лету** к каждому 3-секундному чанку во время обучения (`src/data/augmentation.py`). Исходные FLAC-файлы не изменяются.

**Цепочка применения** (`AugmentationPipeline.__call__`):

```
1. Volume jitter     (p=0.5)  → случайное усиление ±6 dB
2. Speed perturbation (p=0.3) → ресемплинг ×0.9/1.0/1.1, затем обрезка до исходной длины
3. Шум — ОДИН тип на выбор (взаимоисключающие):
     белый шум        (p=0.2)  → Гауссов шум, смешивается по формуле SNR
     розовый шум      (p=0.1)  → 1/f шум, смешивается по формуле SNR
     MUSAN file noise (p=0.5)  → случайный файл из 930 реальных шумов,
                                  обрезается/зациклируется до длины сигнала,
                                  смешивается с заданным SNR (0–15 dB):
                                  noise_rms = clean_rms / 10^(SNR/20)
4. RIR реверберация  (p=0.6)  → случайный IR из 60k файлов OpenSLR,
                                  свёртка через fftconvolve (scipy),
                                  обрезается до исходной длины
5. Канал — ОДИН тип на выбор (взаимоисключающие):
     телефония        (p=0.3)  → bandpass 300–3400 Hz + ресемплинг 8kHz и обратно
     MP3-кодек        (p=0.2)  → MP3 8–32 kbps через ffmpeg, декод обратно в PCM
```

**SNR-смешивание** (`_mix_snr`):
```python
clean_rms = sqrt(mean(clean²))
target_noise_rms = clean_rms / 10^(snr_db / 20)
noise = noise * (target_noise_rms / (noise_rms + 1e-8))
return clean + noise
```

**RIR-свёртка** (`apply_rir`):
```python
rir = rir[peak_idx:]          # обрезаем pre-ringing до пика
rir = rir / max(|rir|)        # нормализуем
reverbed = fftconvolve(clean, rir)[:len(clean)]  # свёртка, обрезка
```

В итоге каждый батч содержит записи с разными случайными комбинациями аугментаций — модель учится быть робастной к шуму, реверберации и канальным искажениям одновременно.

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

### Сабмит 3 — CAM++ zero-shot (Alibaba DAMO, VoxCeleb)

`damo/speech_campplus_sv_en_voxceleb_16k` через ModelScope. GPU batching (256 файлов/батч, ~100 файлов/сек).

| | Public leaderboard P@10 |
|---|---|
| CAM++ zero-shot | **0.4560** |

**+0.070** к SpeechBrain ECAPA. Место: **17/38**.

---

### Сабмит 5 — Ансамбль CAM++ + ECAPA v2 (grid search по alpha)

Конкатенация L2-нормализованных эмбеддингов: `[α × emb_campplus | (1-α) × emb_ecapa_v2]` → 704-dim → L2-normalize → FAISS.

| alpha (CAM++) | alpha (ECAPA v2) | Public P@10 |
|---|---|---|
| 0.8 | 0.2 | 0.4658 |
| **0.7** | **0.3** | **0.4776** ← лучший |

Скрипт: `scripts/run_ensemble.py --grid-search`

---

### Сабмит 6 — ERes2Net zero-shot (Alibaba DAMO, VoxCeleb)

`damo/speech_eres2net_sv_en_voxceleb_16k` (revision `master`) через ModelScope.

**Баг и исправление**: оригинальный `model(padded_batch)` внутри использует `Kaldi.fbank` без батч-поддержки → всегда возвращал `[1, 192]` вместо `[B, 192]`. Итог: 527 правильных эмбеддингов из 134 697, остальные нули. Исправлено добавлением `--torchaudio-frontend` в `run_campplus.py` — GPU-батчевый `torchaudio.MelSpectrogram` подаётся напрямую в `model.embedding_model`.

| | Public leaderboard P@10 |
|---|---|
| ERes2Net zero-shot | **0.4896** |

**+0.034** к CAM++ zero-shot. Новый лучший результат на тот момент.

---

### Сабмит 7 — Fine-tuned CAM++ (10 эпох, MUSAN+RIR, SubcenterArcFace)

Дообучение `damo/speech_campplus_sv_en_voxceleb_16k` на наших train-данных.

**Ключевые детали**:
- Архитектура: `CAMPlusWrapper` (`src/models/campplus_wrapper.py`) — `embedding_model` из ModelScope + GPU-батчевый `torchaudio.MelSpectrogram` вместо последовательного `Kaldi.fbank` (ускорение ~5×)
- Loss: SubcenterArcFace (K=3, margin=0.2, scale=30)
- LR: 1e-4 (fine-tuning, меньше чем при обучении с нуля), cosine decay, warmup 1 эпоха
- Аугментации: MUSAN noise (p=0.5) + RIR (p=0.6) + телефония, кодек, speed, volume
- Batch size: 128, 10 эпох, ~3.1 часа
- Чекпоинт: `weights/campplus_finetune_best.pt` (эп. 9, val P@10=0.9700)
- Инференс: `scripts/run_campplus_ft_infer.py`

| Эпоха | Val P@10 |
|---|---|
| 7 | 0.9684 |
| 8 | 0.9694 |
| **9** | **0.9700** (best) |
| 10 | 0.9695 |

| | Public leaderboard P@10 |
|---|---|
| FT-CAM++ solo | **0.6040** |

**+0.148** к zero-shot CAM++. Огромный прыжок — fine-tuning на данных соревнования с реальными аугментациями дал кратный прирост.

---

### Сабмит 8 — Ансамбль FT-CAM++ + ERes2Net

Конкатенация: `[0.8 × norm(emb_ft) | 0.2 × norm(emb_eres)]` → 704-dim → L2-normalize → FAISS.

FT-CAM++ доминирует (0.6040 vs 0.4896), ERes2Net добавляет разнообразие.

| FT-CAM++ вес | ERes2Net вес | Public P@10 |
|---|---|---|
| 0.8 | 0.2 | 0.6081 |
| **0.7** | **0.3** | **0.6125** ← лучший |
| 0.6 | 0.4 | 0.6125 |

Скрипт: `scripts/run_campplus.py` (ансамбль через `run_ensemble.py`)

---

### Сабмит 4 — WavLM-Base-Plus-SV, zero-shot

| | Public leaderboard P@10 |
|---|---|
| WavLM-Base-Plus-SV v1 (двойная нормализация) | 0.1167 ❌ |
| WavLM-Base-Plus-SV v2 (исправлено) | 0.1149 ❌ |

**Причина провала**: `collate_pad` паддит батч нулями. WavLM без `attention_mask` обрабатывает нули как реальный сигнал → испорченные эмбеддинги. Исправлено: `WavLMEmbedder.embed_batch` теперь обрезает trailing zeros и передаёт переменной длины массивы в feature extractor (который сам создаёт корректный attention_mask).

---

## План на оставшиеся 6 дней (актуальный)

Текущий лучший результат: **0.7322**. Лидер: **0.7379**. Gap: **0.0057**.

### День 1: 5-crop TTA + rerank sweep

**Шаг 1 — 5-crop TTA для обеих дообученных моделей**

Увеличение кропов с 3 до 5 (позиции 0/25/50/75/100%) даёт более стабильный эмбеддинг. Инфраструктура уже готова.

```bash
# CAM++ 5-crop TTA
bash scripts/run.sh scripts/run_tta_infer.py \
    --checkpoint weights/campplus_finetune_best.pt \
    --model-type campplus --n-crops 5 \
    --test-csv "extracted_ата/Для участников/test_public.csv" \
    --cache-name campplus_ft_tta5

# ERes2Net 5-crop TTA
bash scripts/run.sh scripts/run_tta_infer.py \
    --checkpoint weights/eres2net_finetune_best.pt \
    --model-type eres2net --n-crops 5 \
    --test-csv "extracted_data/Для участников/test_public.csv" \
    --cache-name eres2net_ft_tta5
```

**Шаг 2 — rerank sweep поверх 5-crop ансамбля**

```bash
python scripts/run_rerank.py \
    --emb-a embeddings/campplus_ft_tta5.npy \
    --emb-b embeddings/eres2net_ft_tta5.npy \
    --alpha 0.6 --k1 60 --k2 6 --lambda-val 0.1 \
    --test-csv "extracted_data/Для участников/test_public.csv" \
    --output submission_tta5_rerank_k60_l01.csv
```

Перебрать: k1 ∈ {50, 60, 70, 80}, λ ∈ {0.2, 0.15, 0.1}.

---

### Ночь 1–2: Low-LR second-stage fine-tune (безопасный baseline)

Продолжить оба чекпоинта с очень маленьким LR — стандартный second-stage fine-tune. Chunk остаётся 3 сек (не меняем то, что работает).

- **+5 эпох** (эпохи 10–14) поверх stage1 (эпохи 1–9)
- lr=1e-5 (в 10x меньше stage1), warmup_epochs=0
- Чекпоинты: `weights/campplus_stage2_best.pt`, `weights/eres2net_stage2_best.pt`

```bash
# CAM++ stage2
bash scripts/run.sh scripts/run_train_campplus.py \
    --config configs/train_campplus_stage2.yaml \
    --resume --resume-from weights/campplus_finetune_best.pt --reset-best

# ERes2Net stage2 (после CAM++)
bash scripts/run.sh scripts/run_train_eres2net.py \
    --config configs/train_eres2net_stage2.yaml \
    --resume --resume-from weights/eres2net_finetune_best.pt --reset-best
```

Ожидаемый прирост от new checkpoints: **+0.005–0.010**.

---

### Stage3 fine-tune: ещё +10 эпох с lr=5e-6

После stage2 модели ещё не вышли на плато (val P@10 продолжал расти). Запускаем stage3 — продолжение с ещё меньшим LR.

- **+10 эпох** поверх stage2
- lr=5e-6 (в 2x меньше stage2), warmup_epochs=0
- Конфиги: `configs/train_campplus_stage3.yaml`, `configs/train_eres2net_stage3.yaml`
- Чекпоинты: `weights/campplus_finetune_stage3_best.pt`, `weights/eres2net_finetune_stage3_best.pt`

```bash
# CAM++ stage3
bash scripts/run.sh scripts/run_train_campplus.py \
    --config configs/train_campplus_stage3.yaml \
    --resume --resume-from weights/campplus_finetune_stage2_best.pt --reset-best

# ERes2Net stage3 (после CAM++ stage3)
bash scripts/run.sh scripts/run_train_eres2net.py \
    --config configs/train_eres2net_stage3.yaml \
    --resume --resume-from weights/eres2net_finetune_stage2_best.pt --reset-best
```

Следить за прогрессом:
```bash
tail -f results/train_campplus_stage3.log | grep -E "New best|Done|Epoch [0-9]+:.*%"
```

---

### День 3: TTA 5-crop новых чекпоинтов + rerank

Повторить шаг 1 с новыми весами → ансамбль → rerank с лучшими найденными k1/λ.

---

### Ночь 3–4: Эксперимент с chunk=5 сек (параллельный run)

Отдельный эксперимент — **не заменяет** low-LR run, а идёт параллельно. Если даст лучший val P@10 — берём в ансамбль, если нет — не рискуем.

```yaml
# Отдельные конфиги: chunk_s: 5.0, batch_size: 64 (VRAM), lr: 1e-5
```

**Риски**: BatchNorm и AttentiveStatsPool настроены на 3-сек статистики; увеличение чанка — смена режима обучения. Поэтому только как отдельный эксперимент, результат сравниваем с 3-сек вариантом.

---

### День 4–5: 3-model ensemble

Дообучить **ECAPA-TDNN v2** по той же схеме (SubcenterArcFace + MUSAN+RIR, lr=1e-5). У нас уже есть чекпоинт `weights/ecapa_tdnn_v2_realaugment_best.pt`. Добавить как третью ортогональную архитектуру:

```
α·FT-CAM++(TTA5) + β·FT-ERes2Net(TTA5) + γ·FT-ECAPA(TTA5) → rerank
```

Ожидаемый прирост от 3-й архитектуры: **+0.003–0.007**.

> **Важно**: zero-shot модели (CAM++ 0.4560, ERes2Net 0.4896) как 3-й компонент **не добавляем** — тянут ансамбль вниз. Только дообученные.

---

### День 6: Финальный rerank grid search + сабмит

Финальный sweep по гиперпараметрам 3-model ансамбля и сабмит лучшего результата.

---

### Ожидаемые приросты

| Этап | Ожидаемый прирост |
|---|---|
| 5-crop TTA + rerank sweep | +0.002–0.005 |
| Low-LR second-stage fine-tune | +0.005–0.010 |
| chunk=5с эксперимент (если сработает) | +0.002–0.005 |
| 3-model ensemble (ECAPA как 3-я) | +0.003–0.007 |
| **Суммарно** | **+0.010–0.025** |

---

## Перспективы улучшения

### Итоговая таблица результатов

| Модель | Val P@10 | Public P@10 | Примечание |
|---|---|---|---|
| Baseline ONNX | 0.8941 | — | от организаторов |
| ECAPA v1 (синт. аугм.) | 0.9458 | 0.2699 | место 14/29 |
| WavLM zero-shot | — | 0.1149 | баг с padding |
| SpeechBrain ECAPA zero-shot | — | 0.3858 | |
| ECAPA v2 (MUSAN+RIR) | 0.9465 | 0.3717 | |
| CAM++ zero-shot | — | 0.4560 | место 17/38 |
| CAM++ α=0.7 + ECAPA v2 α=0.3 | — | 0.4776 | ансамбль |
| ERes2Net zero-shot | — | 0.4896 | |
| FT-CAM++ solo | 0.9700 | 0.6040 | fine-tune 10 эп. |
| FT-CAM++ 80% + ERes2Net 20% | — | 0.6081 | |
| FT-CAM++ 70% + ERes2Net 30% | — | 0.6125 | |
| FT-CAM++ 60% + ERes2Net 40% | — | 0.6125 | плато |
| TTA-CAM++ 70% + FT-ERes2Net 30% | — | 0.6588 | TTA + FT-ERes2Net |
| TTA-CAM++ 60% + FT-ERes2Net 40% | — | 0.6602 | |
| TTA-CAM++ 50% + FT-ERes2Net 50% | — | 0.6484 | перебор ERes2Net |
| **Rerank(TTA-CAM++ 60% + FT-ERes2Net 40%)** | — | **0.7179** | **k-reciprocal re-ranking** |
| TTA-FT-CAM++ 60% + TTA-FT-ERes2Net 40% + Rerank(k1=20) | — | 0.7208 | оба дообученных, 3-crop TTA |
| TTA-FT-CAM++ 70% + TTA-FT-ERes2Net 30% + Rerank(k1=20) | — | 0.7149 | хуже 60/40 |
| TTA-FT-CAM++ 60% + TTA-FT-ERes2Net 40% + Rerank(k1=30) | — | 0.7272 | |
| TTA-FT-CAM++ 60% + TTA-FT-ERes2Net 40% + Rerank(k1=40, λ=0.3) | — | 0.7303 | |
| TTA-FT-CAM++ 60% + TTA-FT-ERes2Net 40% + Rerank(k1=40, λ=0.2) | — | 0.7310 | |
| TTA-FT-CAM++ 60% + TTA-FT-ERes2Net 40% + Rerank(k1=50, λ=0.2) | — | 0.7322 | 3-crop TTA |
| TTA5-FT-CAM++ 60% + TTA5-FT-ERes2Net 40% + Rerank(k1=70, λ=0.1) | — | 0.7388 | stage1 чекпоинты |
| TTA5-stage3-CAM++ 60% + TTA5-stage3-ERes2Net 40% + Rerank(k1=70, λ=0.1) | — | **0.7413** | stage3, +0.0025 к stage1 |
| TTA5-stage3 + Rerank(k1=80, λ=0.1) | — | 0.7408 | k1=80 хуже k1=70 |
| TTA5-stage3 + Rerank(k1=70, λ=0.15) | — | 0.7411 | λ=0.15 хуже λ=0.1 |
| **→ Оптимум rerank: k1=70, λ=0.1. Далее — 3-model ensemble.** | | | |

#### Динамика дообучения по стадиям

| Модель | Stage1 (10 эп., lr=1e-4) | Stage2 (+5 эп., lr=1e-5) | Stage3 (+10 эп., lr=5e-6) |
|---|---|---|---|
| CAM++ | 0.9700 | 0.9702 | **0.9707** |
| ERes2Net | 0.9673 | 0.9676 | **0.9682** |

Рост замедляется — модели приближаются к плато на val-сплите (чистые данные). Реальный буст виден на leaderboard.

#### Fine-tune ERes2Net (завершено)
- Модель: `damo/speech_eres2net_sv_en_voxceleb_16k`, revision `master`
- Wrapper: `src/models/eres2net_wrapper.py` (GPU torchaudio frontend, тот же паттерн что CAMPlusWrapper)
- Конфиг: `configs/train_eres2net.yaml` — 10 эп., lr=1e-4, SubcenterArcFace K=3, MUSAN+RIR
- Скрипт: `scripts/run_train_eres2net.py`
- Чекпоинт: `weights/eres2net_finetune_best.pt`
- Val P@10: **0.9673**, время: 401 мин

#### TTA (multi-crop)
- Скрипт: `scripts/run_tta_infer.py --n-crops 3`
- Позиции crop'ов: 0%, 50%, 100% от длины аудио → 3 эмбеддинга → усреднение → L2-normalize
- Поддерживает оба типа модели: `--model-type campplus|eres2net`
- Кэш: `embeddings/campplus_ft_tta3.npy` (FT-CAM++), `embeddings/eres2net_ft_tta3.npy` (FT-ERes2Net)
- Оба файла — дообученные модели, 3-crop TTA

#### K-reciprocal Re-ranking
- Скрипт: `scripts/run_rerank.py`
- Реализация: `src/retrieval/reranking.py`
- Параметры лучшего результата: `k1=50, k2=6, lambda=0.2`
- Алгоритм (Zhong et al., CVPR 2017):
  1. FAISS: найти top-`k1*2` соседей для каждого запроса
  2. **K-reciprocal соседи**: j входит в R(i), если i ∈ top-k1(j) — взаимное соседство
  3. **Расширение R**: добавить соседей соседей, если пересечение ≥ 2/3·k1
  4. **Гауссовы веса**: `w(i,j) = exp(-dist(i,j))`, нормировать
  5. **k2-expansion**: усреднить V по top-k2 соседям
  6. **Jaccard distance**: `d_jac(i,j) = 1 - 2·(V_i·V_j) / (|V_i| + |V_j|)`
  7. **Финальный score**: `(1-λ)·d_jaccard + λ·d_cosine`
- Время на 134k векторов: **~2 минуты** (memory-efficient, без N×N матрицы)

```bash
python scripts/run_rerank.py \
    --emb-a embeddings/campplus_ft_tta3.npy \
    --emb-b embeddings/eres2net_ft_tta3.npy \
    --alpha 0.6 \
    --test-csv "extracted_data/Для участников/test_public.csv" \
    --output submission_rerank_tta_both_e4.csv \
    --k1 20 --k2 6 --lambda-val 0.3
```

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
- [x] Ансамбль CAM++ α=0.7 + ECAPA v2 α=0.3 → **0.4776**
- [x] ERes2Net zero-shot (баг torchaudio fix) → **0.4896**
- [x] CAM++ fine-tune (10 эп., GPU MelSpec frontend, SubcenterArcFace) → val P@10=0.9700
- [x] FT-CAM++ инференс → **0.6040** (+0.148 к zero-shot)
- [x] Ансамбль FT-CAM++ 80% + ERes2Net 20% → **0.6081**
- [x] Ансамбль FT-CAM++ 70% + ERes2Net 30% → **0.6125**
- [x] Ансамбль FT-CAM++ 60% + ERes2Net 40% → **0.6125** (плато)
- [x] Fine-tune ERes2Net (10 эп., MUSAN+RIR, SubcenterArcFace) → val P@10=0.9673
- [x] TTA 3-crop для FT-CAM++ → `embeddings/campplus_ft_tta3.npy`
- [x] TTA 3-crop для FT-ERes2Net → `embeddings/eres2net_ft_tta3.npy`
- [x] TTA-CAM++ 60% + FT-ERes2Net 40% → **0.6602**
- [x] K-reciprocal re-ranking (k1=20, k2=6, λ=0.3) поверх 60/40 → **0.7179**
- [x] TTA-both 60/40 + Rerank(k1=20) → **0.7208**
- [x] TTA-both 70/30 + Rerank(k1=20) → 0.7149 (хуже, 60/40 оптимум)
- [x] TTA-FT-CAM++ 60% + TTA-FT-ERes2Net 40% + Rerank(k1=30) → **0.7272**
- [x] TTA-FT-CAM++ 60% + TTA-FT-ERes2Net 40% + Rerank(k1=40, λ=0.3) → **0.7303**
- [x] TTA-FT-CAM++ 60% + TTA-FT-ERes2Net 40% + Rerank(k1=40, λ=0.2) → **0.7310**
- [x] TTA-FT-CAM++ 60% + TTA-FT-ERes2Net 40% + Rerank(k1=50, λ=0.2) → **0.7322**
- [x] TTA5-FT-CAM++ 60% + TTA5-FT-ERes2Net 40% + Rerank(k1=70, λ=0.1) → **0.7388** (stage1)
- [x] TTA5-stage3-CAM++ 60% + TTA5-stage3-ERes2Net 40% + Rerank(k1=70, λ=0.1) → **0.7413** ← лучший
- [x] Оптимум rerank найден: k1=70, λ=0.1. k1=80 и λ=0.15 хуже.
- [ ] 3-model ensemble (+ FT-ECAPA как 3-я модель) → 🔜
- [x] CAM++ stage2 (lr=1e-5, эп. 10–14) → val P@10 **0.9702** (+0.0002 к stage1)
- [x] ERes2Net stage2 (lr=1e-5, эп. 10–14) → val P@10 **0.9676** (+0.0003 к stage1)
- [x] CAM++ stage3 (lr=5e-6, эп. 14–23) → val P@10 **0.9707** (+0.0005 к stage2)
- [x] ERes2Net stage3 (lr=5e-6, эп. 15–24) → val P@10 **0.9682** (+0.0006 к stage2)
- [ ] TTA 5-crop stage3 чекпоинтов → 🟡 в процессе
- [ ] Rerank sweep поверх stage3 TTA → ⏳
- [ ] Сабмит лучшего результата
