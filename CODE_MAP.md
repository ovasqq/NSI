# Карта кода

Короткий указатель: какая папка за что отвечает. Подробное описание каждого файла — в `README.md` соответствующей папки.

| Папка | Этап | Назначение | Подробно |
|---|---|---|---|
| [data/](data/) | 0 | Скачивание и разбор графов данных G; кэш сырых архивов и сохранённые наборы запросов | [data/README.md](data/README.md) |
| [search/](search/) | 1 | Классический точный поиск: граф, фильтры кандидатов, порядок φ, перебор с хуком переупорядочивания | [search/README.md](search/README.md) |
| [queries/](queries/) | 2 | Генерация запросов случайным блужданием, хранение наборов, эйлерова сериализация для модели | [queries/README.md](queries/README.md) |
| [experiments/](experiments/) | 2 | Скрипты замеров (FMS), результаты и логи | [experiments/README.md](experiments/README.md) |
| [models/](models/) | 3 | Модель NeuGN (GCN + Transformer), формат входа, хук для перебора | [models/README.md](models/README.md) |
| [train/](train/) | 3 | Генерация обучающих примеров случайным блужданием и цикл обучения | [train/README.md](train/README.md) |
| [tests/](tests/) | 0–3 | Проверки каждого этапа (сверка с networkx, инварианты, модель) | [tests/README.md](tests/README.md) |
| `checkpoints/` | 3 | Обученные модели: `checkpoints/<graph>/<run>/{config.json, log.jsonl, model.pt}` (в `.gitignore`) | [train/README.md](train/README.md) |

## Как данные проходят через код

```
data/loaders.py ──► search.graph.Graph (G)
                         │
queries/random_walk.py ──┤  Query = (Graph Q, origin)
queries/generate.py  ────┴─► data/queries/<graph>/size<N>_<density>.json   (queries/storage.py)
                                         │
experiments/fms.py ── для каждого Q: ────┘
     search/filter.py   C(u)            кандидаты для каждой вершины Q
     search/order.py    φ               порядок сопоставления
     search/enumerate.py                перебор; хук reorder ◄── models/hook.py (NeuGN)
                         └─► experiments/results/<tag>.json

Обучение NeuGN:
data/loaders.py ─► train/dataset.py (блуждание → Q → эйлерова посл. → маски)
                     └─ models/encoding.py (токены) ─► models/neugn.py ◄─ train/train_neugn.py
                                                          └─► checkpoints/<graph>/<run>/model.pt
```

## Корневые файлы

| Файл | Что в нём |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Полный контекст проекта: задача, статья NeuGN, наши идеи, план этапов |
| [DECISIONS.md](DECISIONS.md) | Все решения, которых нет в статье (что выбрано и почему) |
| [GLOSSARY.md](GLOSSARY.md) | Терминология: задача, алгоритм, метрики (FMS и др.), NeuGN, обучение |
| [PROGRESS_1.md](PROGRESS_1.md), [PROGRESS_2.md](PROGRESS_2.md), [PROGRESS_3.md](PROGRESS_3.md) | Отчёты о ходе работ по этапам |
| [ORACLE_EXPERIMENT.md](ORACLE_EXPERIMENT.md) | Оракульный эксперимент (этап 4, go/no-go): истинные p/c, что и как сделано, результаты и вердикт |
| [PHI_EXPERIMENT.md](PHI_EXPERIMENT.md) | Замер порядка φ (этап 4b): есть ли запас в выборе порядка вершин (GQL/RI/случайный) |
| [IDEA2_DESIGN.md](IDEA2_DESIGN.md), [IDEA2_RESULTS.md](IDEA2_RESULTS.md) | Идея 2 (индуктивный скоринг): дизайн и результаты (Hamster, глубина, перенос на rewired) |
| [PRESENTATION_BRIEF_1.md](PRESENTATION_BRIEF_1.md) | Итог для презентации: воспроизведение NeuGN |
| [PRESENTATION_BRIEF_2.md](PRESENTATION_BRIEF_2.md) | Итог для презентации: ранжирование по p/c без дообучения (отрицательный результат) |
| [pyproject.toml](pyproject.toml) | Зависимости и настройки pytest (`pythonpath = ["."]`, поэтому импорты вида `from search.graph import Graph`) |
| `.gitignore` | Исключает `.venv/`, `data/raw/`, `checkpoints/`, `experiments/logs/` |

## Запуск

```bash
.venv/bin/python -m pytest                       # все тесты
.venv/bin/python -m queries.generate             # наборы запросов (≈1 мин)
.venv/bin/python -m experiments.fms --graph hamster --size 20 --density dense
.venv/bin/python -m train.train_neugn --graph hamster --run paper   # ≈1 ч на MPS
.venv/bin/python -m experiments.fms --graph hamster --size 20 --density dense \
    --method neugn --checkpoint checkpoints/hamster/paper/model.pt
.venv/bin/python -m experiments.oracle --graph hamster --sizes 8 10 --densities dense sparse \
    --count 100 --checkpoint checkpoints/hamster/paper/model.pt   # оракул (этап 4)
.venv/bin/python -m experiments.phi --graph hamster --sizes 8 10 --densities dense sparse   # замер φ
```
