# experiments/ — замеры

Скрипты, которые прогоняют методы на сохранённых наборах запросов и считают метрики. Каждый запуск пишет JSON со всей конфигурацией, метаданными набора, сводкой и записью по каждому запросу, чтобы результат можно было перепроверить и воспроизвести.

## Файлы

### [fms.py](fms.py) — протокол FMS (First Match Steps)

```bash
.venv/bin/python -m experiments.fms --graph hamster --size 20 --density dense \
    [--method baseline|random|neugn] [--filter nlf|ldf] [--order gql|ri] \
    [--k 1] [--budget 1000000] [--limit N] [--progress] [--seed 0] [--tag SUFFIX] \
    [--checkpoint checkpoints/<graph>/<run>/model.pt] [--max-depth 10] [--device cpu] \
    [--pruning none|arc|domains|motifs] \
    [--pc-depth 3] [--probes 4] [--alpha 1.0] [--query-set path.json]
```

- **`run_query(q, g, factory, *, filter_name, order_name, k, budget, pruning) -> dict`** — один запрос: фильтр → порядок → `factory(q, g, cand, order)` создаёт хук → перебор до k вложений или исчерпания бюджета шагов. Возвращает:
  - `fms` — шаг первого вложения. `None` при `complete=true` означает, что поиск доказал отсутствие вложений. `None` при `complete=false` означает, что вложение не найдено за бюджет;
  - `match_steps` — шаги 1-го … k-го вложений;
  - `steps`, `complete`;
  - `filter_seconds`, `search_seconds`, `total_seconds` — время подготовки запроса, поиска и их сумма;
  - при `--pruning` в одном из [режимов проверок](../search/README.md#pruningpy--проверки-остаточной-задачи) пишутся `pruning_calls`, `pruning_seconds`, `motif_checks` и `pruning_cuts` по причинам. Внутренняя работа проверок входит во время поиска. Бюджет по-прежнему ограничивает назначения DFS;
  - для хуков со спусками: `probe_steps` и `fms_with_probes` = FMS + шаги спусков (`None`, если сумма больше бюджета). Бюджет ограничивает только шаги самого перебора.
- **`censored_median(values)`** — медиана, где `None` считается +∞. Она точна, пока цензурировано меньше половины запросов; иначе получается ∞ и печатается как «> budget».
- **`summarize(records, k, budget)`**:
  - `solved` — сколько запросов нашли первое вложение;
  - `median_fms` (основная метрика статьи);
  - `median_steps_to_k`;
  - `exhausted_without_k` — дерево обойдено целиком, а k вложений нет (это не провал бюджета: вложений просто меньше k);
  - `steps_per_second`;
  - суммарное `total_seconds` и, при включённых проверках, их суммарная статистика;
  - для методов со спусками: `solved_with_probes`, `median_fms_with_probes`.
- `--query-set` — взять набор запросов из файла (например, валидационный `data/queries/hamster_val/…`) вместо стандартного.
- `--progress` — показывать прогресс по завершённым запросам. Установить `tqdm` в окружении запуска можно через `python -m pip install tqdm`.
- **Методы** (`make_hook_factory(args)` возвращает функцию «запрос → хук»; хук создаётся заново для каждого запроса):
  - `baseline` — без хука, кандидаты по возрастанию id;
  - `random` — равновероятная случайная перестановка на каждой глубине (sanity check: NeuGN должен быть лучше и того и другого);
  - `knuth` — только 1/ĉ: `models.pc_hook.PCReorder` без модели (`--pc-depth`, `--probes`);
  - `neugn_pc` — p̂/ĉ: NeuGN + оценка Кнута на глубинах `< --pc-depth`, вес стоимости `--alpha`, ниже — обычный NeuGN;
  - `neugn` — `models.hook.NeuGNReorder` с моделью из `--checkpoint`, переупорядочивание на глубинах `< --max-depth`, инференс на `--device`. В запись запроса дополнительно попадают `model_calls` и `model_seconds`.
- **Выход:** `experiments/results/<graph>_size<N>_<density>_<method>_<filter>_<order>_k<k>[_pruning_<mode>][_<tag>].json` с ключами `config`, `query_set`, `summary`, `records`; в stdout — одна строка сводки.

### `__init__.py`

Пустой, нужен для запуска через `python -m experiments.fms`.

## Папки

| Папка | Что лежит |
|---|---|
| `results/` | JSON-результаты `fms.py` |
| `logs/` | stdout фоновых запусков |
