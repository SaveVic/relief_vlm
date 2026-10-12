# Can Vision–Language Models Read Stone?

Code for the experiment in *Can Vision–Language Models Read Stone? Enhancement
Helps, but Counting Is the Bottleneck*. It extracts ground truth from
colour-coded segmentation masks of temple relief panels, runs a grid of
vision–language model calls through Replicate under several input conditions
(photo, photo + depth, photo + raking-light fusion + cavity map), and scores the
structured output against the annotation.

Every stage reads its inputs from disk and writes its outputs to disk, so a stage
can be rerun on its own or resumed after an interruption.

> **Language note.** The code is in English. The system prompt, the per-condition
> instructions, the JSON Schema field names (`objek`, `kategori`, `keyakinan`, ...)
> and the repair feedback sent to the model are in Indonesian on purpose: they are
> exactly what produced the reported results, and translating them would change
> the experiment. Condition IDs such as `RGB_SAJA` ("RGB only") are kept for the
> same reason, since they appear in every stored record.

> **Data.** The panel photographs and masks may not be redistributed under the
> field permit, so this repository contains code only.

---

## Install

```bash
git clone <repo-url> relief-vlm
cd relief-vlm

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[all]"            # or: pip install -r requirements.txt
cp .env.example .env               # then fill in REPLICATE_API_TOKEN and paths
```

Python 3.10 or newer.

## Configuration and paths

Every path comes from configuration, never from the working directory, so the
CLI and all scripts can be run from anywhere. Priority:

```
CLI flag  >  environment variable / .env  >  configs/default.yaml  >  defaults in config.py
```

| variable | meaning | default |
|---|---|---|
| `REPLICATE_API_TOKEN` | Replicate API token | — |
| `RELIEF_DATA_ROOT` | dataset, one folder per panel | `data/dataset` |
| `RELIEF_OUTPUT_ROOT` | ground truth, split, manifests, runs, evaluation | `outputs` |
| `RELIEF_FIGURES_DIR` | figures written by scripts | `<output root>/figures` |
| `RELIEF_CONFIG` | alternative YAML config | `configs/default.yaml` |
| `RELIEF_ENV_FILE` | alternative env file (set in the shell) | `./.env` |

`.env` is read automatically and is git-ignored. See `.env.example` for the
remaining overrides (seed, temperature, repair budget, upload mode).

Expected dataset layout:

```
<RELIEF_DATA_ROOT>/
├── Boro_01/
│   ├── rgb.png      photograph
│   ├── depth.png    monocular depth map
│   ├── fusion.png   derived: raking-light fusion image
│   ├── cavity.png   derived: cavity map
│   └── seg2.png     colour-coded instance mask
├── Boro_02/
└── ...
```

---

## Pipeline

```bash
relief fetch --zip-url "<Drive zip link>"   # download the dataset (optional)
relief verify                               # check every panel folder is complete
relief gt                                   # layer-1 ground truth from the masks
relief split                                # frozen dev/test split
relief models                               # check endpoint input names
relief plan --plan dev                      # manifest + token/cost forecast
relief smoke                                # one test call per model
relief run --plan dev                       # run the grid (uses credits)
relief usage                                # measured tokens, time, cost
relief evaluate                             # scores, paired tests, LaTeX tables
```

Without installing, use `python -m relief <command>`. Only `smoke` and `run`
spend credits. `run` accepts `--max-tokens-total N` as a budget guard.

### Analysis scripts

Run after `relief evaluate`. All read their paths from the configuration above.

| script | purpose |
|---|---|
| `scripts/00_split_dev_runs.py` | move development-plan records out of the main run folder |
| `scripts/01_replicate_agreement.py` | replicate noise and its share of the interval width |
| `scripts/02_revalidate.py` | re-validate stored outputs against the schema without new API calls |
| `scripts/03_primary_tests.py` | pre-registered primary tests with Holm correction |
| `scripts/04_descriptive_findings.py` | listing ratio, omission vs. mismatch, calibration, status breakdown |
| `scripts/05_reset_errors.py` | remove records whose API call failed so `run` retries them |
| `scripts/06_fusion_vs_depth.py` | `RGB_FUSION_CAVITY` vs. `RGB_DEPTH` (exploratory) |
| `scripts/07_check_panel_names.py` | check panel folder names (read-only) |
| `scripts/08_example_figure.py` | best/worst example figure for the paper |
| `scripts/run_tests.py` | run the test suite without pytest |

Scripts that change files are dry runs by default: `00_split_dev_runs.py --move`,
`02_revalidate.py --write` and `05_reset_errors.py --delete` apply the change.
Earlier (Indonesian) names, as referenced in `PRAREGISTRASI.md`:
`00_pisah_dev` → `00_split_dev_runs`, `01_agreement` → `01_replicate_agreement`,
`03_uji_utama` → `03_primary_tests`, `04_temuan_deskriptif` →
`04_descriptive_findings`, `05_reset_error` → `05_reset_errors`,
`06_fc_vs_depth` → `06_fusion_vs_depth`, `07_cek_nama_panel` →
`07_check_panel_names`, `08_figure_contoh` → `08_example_figure`,
`jalankan_uji` → `run_tests`.

---

## What is stored for every API call

Each manifest cell gets its own folder, and each attempt (first call plus any
repair) is stored in three layers:

```
<output root>/runs/<model>/<cell_id>/
├── record.json                  status, usage, trace of all attempts (written last)
├── request.json                 system prompt, user prompt, image hashes
├── attempt-01.raw.txt           exact model text
├── attempt-01.prediction.json   full Replicate prediction (metrics, logs, status)
├── attempt-01.parsed.json       json.loads result        (if parsable)
├── attempt-01.validated.json    schema-validated result  (if valid)
├── attempt-01.error.txt         parse / validation error (if any)
└── attempt-02.*                 repair attempt, if any
```

| status | meaning |
|---|---|
| `ok` | JSON valid against the schema |
| `schema_failed` | JSON parses but violates the schema; scored from the parsed content and flagged |
| `json_failed` | text returned, but not JSON; scored as zero recall |
| `error` | the API call itself failed |

`record.json` is written last and marks a cell as done, so rerunning `run`
resumes exactly where it stopped.

### Token usage

Replicate endpoints name their metrics differently (`input_token_count`,
`input_tokens`, `prompt_tokens`, ...). `relief/usage.py` tries known names
first, falls back to pattern matching, and finally parses the logs. The raw
metrics are always saved in `attempt-*.prediction.json`, so counts can be
recomputed without new calls. `source: "none"` means the count could not be read
(not zero) and is reported separately.

---

## Input conditions and plans

| condition | images | role |
|---|---|---|
| `RGB_SAJA` | RGB | baseline |
| `RGB_DEPTH` | RGB + DEPTH | main |
| `RGB_FUSION_CAVITY` | RGB + FUSION + CAVITY | main |
| `RGB_X3` | RGB ×3 | control: image count |
| `RGB_FUSION` | RGB + FUSION | control: component |
| `RGB_CAVITY` | RGB + CAVITY | control: component |

| plan | conditions | split | replicates |
|---|---|---|---|
| `dev` | three main conditions | dev (10 panels) | 3 |
| `main` | three main conditions | test (46 panels) | 2 |
| `abl_imagecount` | `RGB_X3` | test | 1 |
| `abl_components` | `RGB_FUSION`, `RGB_CAVITY` | test | 1 |
| `abl_resolution` | two conditions at 768 and 3072 px | test | 1 |

All reported numbers come from `main`. The primary comparison was fixed before
the main run; see [`PRAREGISTRASI.md`](PRAREGISTRASI.md) (pre-registration, kept
in its original Indonesian as the dated record).

## Methodological notes

- **The split is frozen.** `seed` in `configs/default.yaml` is written to
  `split.json`; changing it after seeing results breaks independence.
- **One repair budget for all models.** `max_repairs` comes from config, not
  per model; extra attempts for weaker models would confound the comparison.
- **Check `relief models` before adding a model.** A wrong input key raises no
  error: the endpoint silently drops the images or the system prompt. Also check
  the output-length limit; a full record needs about 4–5k tokens.
- **Parse failure is a result, not a bug.** It is reported per model alongside
  `unusable_rate` and `partial_rate`.

---

## Layout

```
relief-vlm/
├── configs/default.yaml
├── scripts/                   analysis scripts (see above)
├── src/relief/
│   ├── cli.py                 subcommands
│   ├── config.py              YAML + env + CLI configuration
│   ├── fetch_data.py          dataset download + verification
│   ├── prompts.py             system prompt + condition registry (Indonesian text)
│   ├── schema.py              output schema, Pydantic (Indonesian field names)
│   ├── models.py              model specs and input-name mapping
│   ├── client.py              Replicate wrapper (uploads, polling, retries)
│   ├── analyze.py             one logical call -> three stored layers
│   ├── artifacts.py           per-attempt file writer and loader
│   ├── usage.py               token normalisation
│   ├── pricing.py             measured cost + forecast
│   ├── imaging.py             encoding + fusion/cavity enhancement
│   ├── manifest.py            split, plans, cells
│   ├── runner.py              execution + resume
│   ├── reporting.py           usage tables + CSV
│   ├── gt.py                  ground truth from masks
│   ├── evaluate.py            scoring, paired tests, LaTeX tables
│   └── viz.py                 display helpers
└── tests/
```

## Tests

```bash
python scripts/run_tests.py      # no pytest needed
# or, with dev extras installed:
pytest -q
```

The tests use a fake client: no network calls and no credits.
