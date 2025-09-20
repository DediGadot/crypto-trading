Done. I created a ready-to-commit **NEXT.md** with a step-by-step, LLM-friendly plan to take the repo to a SOTA pipeline.

[Download NEXT.md](sandbox:/mnt/data/NEXT.md)

> Note: I aligned the plan with your README’s current status (e.g., **SMA backtest mismatch** and the “**50/100** foundation” score), so Milestone 1 focuses on backtest correctness and no-lookahead before anything else. ([GitHub][1])

---

### The file contents (for quick review)

# NEXT.md — SOTA Upgrade Implementation Plan

> Audience: a coding LLM (and human maintainers) implementing a robust, execution-aware, leakage-free crypto trading pipeline in this repo.
>
> Guiding lenses:
>
> * **Quant (robustness-first):** leakage control, cost & execution realism, regime awareness, P\&L-aligned labels, out-of-sample validation.
> * **Systems (simplicity-first):** small modules, clear interfaces, strong typing & tests, reproducible tooling, observable runtime.

---

## 0) Ground rules (do these first)

* **Branching:** create a feature branch per milestone: `feat/<milestone>-<slug>`.
* **Style/tooling:** add and enforce:

  * `ruff`, `black`, `isort`, `mypy --strict`, `pytest`, `hypothesis`.
  * `pre-commit` with hooks for the above.
* **Project layout:** keep existing code runnable, but migrate to a clearer package boundary in *small PRs*:

  ```text
  safla/
    data/          # loaders, validators, cache (Parquet/Arrow), CCXT adapters
    features/      # transforms, indicators, orderbook-derived features
    labels/        # triple-barrier, meta-labeling, label-times
    models/        # fit/predict, CV wrappers, calibration
    backtest/      # vectorized + event-driven adapters, cost & fill models
    exec/          # live routes: REST/WebSocket, OMS, retry/backoff
    risk/          # sizing/vol-target, Kelly cap, drawdown guards
    portfolio/     # MVO/HRP/constraints, discrete allocation
    metrics/       # PSR, turnover, capacity, hit rate decomposition
    cli/           # typer-based CLI entrypoints
    utils/         # timezones, calendars, configs, logging
  ```
* **Config:** use `pydantic-settings` (env-first). Keep secrets in `.env` and **never** in YAML.
* **Determinism:** seed all stochastic components; pin library versions; prefer UTC tz-aware timestamps.
* **Testing policy:** each public function has at least one unit test; each milestone adds integration tests and a reproducible demo script.

---

## Milestone 1 — Backtester correctness (fix SMA + no-lookahead)

**Goal:** a bar-based backtester that never uses information from the future and strictly trades on **t+1**.

### Tasks

1. Implement `safla/backtest/alignment.py`

   * `def shift_for_decision(series: pd.Series) -> pd.Series:` (always `.shift(1)`).
   * `def warmup_mask(df: pd.DataFrame, lookbacks: dict[str, int]) -> pd.Series:` (False until all lookbacks valid).
2. Implement `safla/backtest/signals.py`

   * `def sma_crossover(close: pd.Series, fast:int, slow:int) -> tuple[pd.Series, pd.Series]:`

     * Cross logic computed at `t`, **decisions applied at `t+1`**.
3. Patch the existing SMA strategy to use the above helpers.
4. Enforce **right alignment** for *all* decision inputs in the backtester.
5. Add tests in `tests/test_backtest_alignment.py`:

   * **No-lookahead:** synthetic series with known cross → assert trade at `t+1`, *not* at `t`.
   * **Warmup:** no trades before both MAs valid.
   * **Idempotence:** running the backtest twice yields identical trades/P\&L.
6. Add a smoke test script: `scripts/smoke_sma.py` to reproduce expected trades on a toy series.

### Acceptance

* `pytest -q` passes.
* A failing test from the original SMA mismatch is now green.
* Smoke script prints deterministic entry/exit timestamps and P\&L.

---

## Milestone 2 — Purged & embargoed walk-forward *strategy* evaluation

**Goal:** extend leakage control from model CV into strategy backtests.

### Tasks

1. Add `safla/backtest/splitting.py`:

   * `class PurgedWalkForward:` yielding (train, test) windows with `purge="24h"` and `embargo="24h"`.
   * Emits **label times** for purge (if available from labels module).
2. Integrate with the backtester: for each split → fit params (if any) on train → evaluate on test → aggregate metrics.
3. CLI: `safla backtest walk-forward --purge 24h --embargo 24h --folds 6`.
4. Tests in `tests/test_wf_purged.py`:

   * Overlapping labels do **not** leak (verify by construction with deterministic horizons).
   * Report includes per-fold and aggregate metrics.

### Acceptance

* Walk-forward report shows per-fold Sharpe, drawdown, turnover.
* A run with synthetic labels demonstrates no leakage when horizons overlap.

---

## Milestone 3 — Costs, slippage & fills

**Goal:** realistic P\&L with configurable fee schedules and slippage/impact.

### Tasks

1. `safla/backtest/costs.py`:

   * `class FeeSchedule(maker_bps: float, taker_bps: float, min_fee: float|None = None)`.
   * `def apply_fees(trades_df: pd.DataFrame, fee_schedule: FeeSchedule) -> pd.Series`.
2. `safla/backtest/slippage.py`:

   * `def adv_slippage(qty: float, adv: float, k: float, noise: float=0.0) -> float`.
   * Provide simple **TWAP/VWAP** execution impact hooks.
3. `safla/backtest/fills.py`:

   * Bar-close fills default; partial fills optional.
4. Wire into P\&L pipeline; surface config via CLI/`config.yaml`.

### Tests

* `tests/test_costs_fills.py`: maker/taker fees, min fee handling, slippage proportional to trade size, partial fill math.

### Acceptance

* Backtest report now includes **gross vs net** returns and **turnover**.

---

## Milestone 4 — Labels & sizing that map to P\&L

**Goal:** replace fixed-horizon targets with **triple-barrier** + meta-labeling; add volatility targeting.

### Tasks

1. `safla/labels/triple_barrier.py`:

   * `triple_barrier_labels(prices, pt, sl, vertical, vol=None) -> DataFrame[{'label','t1'}]`.
   * Vol-scaled barriers (estimate `vol` from rolling σ).
2. `safla/models/meta.py`:

   * Meta-labeling wrapper gating primary entries by predicted probability × edge **> costs**.
3. `safla/risk/sizing.py`:

   * `vol_target_position(sig: pd.Series, target_ann_vol: float) -> pd.Series`.
4. Calibrate decision thresholds on *net* objective (after costs).

### Tests

* `tests/test_triple_barrier.py`: barrier hit ordering, vertical timeout, reproducible labels.
* `tests/test_sizing.py`: position scaling hits target σ on synthetic GBM.

### Acceptance

* Training/validation scripts produce label distribution stats.
* Walk-forward backtests exhibit lower variance due to vol-targeting.

---

## Milestone 5 — Metrics & reporting

**Goal:** honest performance assessment beyond raw Sharpe.

### Tasks

1. `safla/metrics/perf.py`:

   * `probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)`.
   * Capacity curve (PnL vs position size), hit rate, avg win/loss, exposure vs BTC.
2. HTML/Markdown report generator in `safla/metrics/report.py`.

### Tests

* Closed-form PSR sanity checks on synthetic returns.
* Capacity decreases as trading size increases (synthetic impact).

### Acceptance

* Reports show PSR with confidence levels and capacity plot images saved under `artifacts/`.

---

## Milestone 6 — Fast research loop (vectorbt)

**Goal:** iterate factors & params quickly, then promote survivors to the event-driven engine.

### Tasks

1. Optional dependency: `vectorbt` (guarded import).
2. `safla/backtest/vectorized_adapter.py`:

   * `from_signals(entries, exits, fees, slippage)` grid sweeps.
3. Parity tests vs our bar backtester for SMA strategy (within tolerance given costs).

### Acceptance

* Benchmark run over a small parameter grid completes < 10s on 1k bars and matches our engine within ±ε P\&L.

---

## Milestone 7 — Event-driven adapter (paper/live parity)

**Goal:** path to realistic execution and future live trading without rewriting strategies.

### Tasks

1. Add `safla/backtest/event_adapter.py`:

   * Adapter interface around an event-driven engine (keep internal or integrate a third-party like Nautilus Trader if adopted).
2. Implement simple **TWAP** execution with latency and partial fills; model dropped ticks and retries.
3. Parity test (qualitative) against vectorized engine under same fees/slippage.

### Acceptance

* Demo script runs event-driven backtest for SMA strategy and produces a report comparable to vectorized results.

---

## Milestone 8 — Order book ingestion & microstructure features

**Goal:** add short-horizon predictive features tailored to crypto microstructure.

### Tasks

1. `safla/data/websocket.py`: Binance (or CCXT Pro) **diff-depth** stream with official snapshot+resync logic, backoff/reconnect.
2. `safla/features/orderbook.py`:

   * `orderbook_imbalance(bid_sizes, ask_sizes)`
   * `microprice(bid_px, bid_sz, ask_px, ask_sz)`
3. Persist to Parquet with partitioning (symbol/date).

### Tests

* Resync logic unit test (simulate out-of-order/lost updates).
* Feature calculations vs toy books.

### Acceptance

* Pipeline builds OB features for a short historical window and shows their distribution/IC vs future returns.

---

## Milestone 9 — Portfolio & risk enhancements

**Goal:** practical portfolio construction with turnover & discrete allocation.

### Tasks

1. Add turnover penalty and discrete lot sizing to PyPortfolioOpt calls.
2. Rebalance cadence control; borrow & funding cost awareness (for perps).
3. Kelly-fraction cap with drawdown guard.

### Tests

* Turnover penalty reduces trades in optimizer outputs.
* Discrete allocation matches lot size constraints.

### Acceptance

* Portfolio backtests show net P\&L improvements after costs with controlled turnover.

---

## Milestone 10 — Data engineering & validation

**Goal:** reproducible data lake & schema guarantees.

### Tasks

1. `safla/data/ccxt_loader.py`: idempotent fetchers; store raw/clean/feature layers in Parquet+Arrow.
2. `safla/data/validation.py`: Pandera schemas for OHLCV, features, and labels; outlier checks (MAD), NaN handling.
3. Timezone normalization to UTC; column naming conventions.

### Tests

* Schema tests fail on malformed CSVs; loader is idempotent.
* Round-trip (load → transform → save → load) preserves dtypes and index.

### Acceptance

* `make data-demo` builds a tiny lake under `./data/` and validates successfully.

---

## Milestone 11 — CLI, config & observability

**Goal:** one‐line commands and production-grade logs/metrics.

### Tasks

1. CLI via Typer: `safla` with subcommands:

   ```bash
   safla data fetch --symbols BTC/USDT ETH/USDT --tf 1h --since 2023-01-01
   safla features build --symbols BTC/USDT --include ob_imbalance microprice
   safla labels triple-barrier --h 24h --pt 2*vol --sl 1*vol --vertical 3d
   safla backtest walk-forward --strategy sma --purge 24h --embargo 24h
   safla research grid --strategy sma --grid "fast=3..20,slow=10..80"
   safla report show --run-id <id>
   ```
2. Logging: `structlog` JSON logs; add run IDs; Prometheus metrics hooks (latency, dropped ticks, order rejects); Sentry optional.

### Acceptance

* Each command runs end-to-end with helpful logs; reports saved under `artifacts/<run-id>/`.

---

## Milestone 12 — CI/CD & developer experience

**Goal:** prevent regressions and ensure reproducibility.

### Tasks

1. GitHub Actions:

   * Lint/type/test jobs.
   * “Determinism” job: run a short backtest twice, assert identical trade stream.
2. Pre-commit in CI; cache Python deps.
3. Optional: `devcontainer.json` for VS Code + `Makefile` targets:

   ```makefile
   setup:  # create venv, install, pre-commit install
   test:   # run pytest
   lint:   # ruff + mypy
   demo:   # run a small end-to-end pipeline
   ```

### Acceptance

* Green pipeline; reproducible demo artifacts attached as workflow artifacts.

---

## File creation checklist (per milestone)

* [ ] New module under `safla/...` with docstrings & type hints.
* [ ] Unit tests under `tests/`.
* [ ] Update CLI/help where relevant.
* [ ] Add/adjust config keys with defaults.
* [ ] Add a short demo script under `scripts/` (if helpful).
* [ ] Update `README` snippets and `CHANGELOG.md`.

---

## PR template (copy into `.github/pull_request_template.md`)

```markdown
## What
Short description + link to milestone.

## Why
Correctness/perf/feature/etc. Include risk/assumptions.

## How
Key implementation notes. Config keys added/changed.

## Tests
- [ ] Unit: ...
- [ ] Integration: ...
- [ ] Determinism: ...
- [ ] Performance: ...

## Acceptance Evidence
Paste CLI output snippets and metrics.

## Checklist
- [ ] Docs updated
- [ ] Config defaults sane
- [ ] Backward compatible (or migration notes)
```

---

## “Do/Don’t” for the coding LLM

**Do**

* Prefer small, testable functions with explicit inputs/outputs.
* Align all decision signals with `.shift(1)`; enforce warmup masks.
* Measure and report **net** performance (after fees/slippage).
* Seed randomness; keep runs deterministic unless explicitly randomized.

**Don’t**

* Don’t change live-facing code and backtest code in the same PR.
* Don’t add new dependencies without updating `requirements.txt` and pinning.
* Don’t compute features on the same bar you trade.

---

## Definition of Done (SOTA pipeline)

* Backtests are **leakage-free**, **purged/embargoed**, and **deterministic**.
* Costs, slippage, and fills are modeled; reports show **gross vs net**.
* Labels are **triple-barrier** + optional **meta-labeling**; positions are **vol-targeted**.
* Metrics include **PSR** and **capacity**; parity established between vectorized and event-driven paths.
* Data lake reproducible with validated schemas; CLI drives end-to-end runs.
* CI enforces style, typing, tests, and determinism.

---


