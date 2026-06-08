#!/usr/bin/env python3
"""Submission-oriented validation workspace for ODE mu inputs.

This script adds checks that are useful for a paper-methodology discussion but
do not require rerunning the expensive CNN/LSTM training stack:

1. locked-candidate metrics on the existing OOS handoff grid;
2. fold-boundary purged sensitivity, dropping the first horizon days in each
   60-day test fold;
3. stationary block-bootstrap CIs for selected locked comparisons;
4. return-scale calibration sanity metrics.

It is intentionally dependency-free so the public GitHub repo can reproduce the
validation tables without rebuilding the deep-learning environment.
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).parent
HANDOFF = ROOT / "outputs" / "ode_handoff"
MU_DIR = HANDOFF / "02_mu_inputs"
OUT_DIR = HANDOFF / "06_mu_submission_validation"
SMOKE_SUMMARY = OUT_DIR / "purged_retraining_smoke" / "smoke_summary.json"

FINAL_WIDE = MU_DIR / "final_mu_inputs_wide.csv"
SELECTED_CALIBRATED = MU_DIR / "selected_mu_input_calibrated.csv"

HORIZON = 20
FOLD_TEST_DAYS = 60
TOP_K = 2
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_BLOCK_LENGTH = HORIZON
SEED = 42

CANDIDATES = {
    "mu_rank_baseline": "mu_signal_mu_rank_baseline",
    "mu_image_factor_rank": "mu_signal_mu_image_factor_rank",
    "mu_image_factor_strict_rank": "mu_signal_mu_image_factor_strict_rank",
    "mu_image_factor_balanced": "mu_signal_mu_image_factor_balanced",
    "mu_sharpe_baseline": "mu_signal_mu_sharpe_baseline",
}

LOCKED_COMPARISONS = [
    ("image_factor_rank_vs_rank_baseline", "mu_rank_baseline", "mu_image_factor_rank"),
    ("strict_rank_vs_rank_baseline", "mu_rank_baseline", "mu_image_factor_strict_rank"),
    ("balanced_vs_sharpe_baseline", "mu_sharpe_baseline", "mu_image_factor_balanced"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def to_float(value: str) -> float:
    if value == "" or value.lower() == "nan":
        return float("nan")
    return float(value)


def is_finite(value: float) -> bool:
    return math.isfinite(value)


def rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 1e-18 or vy <= 1e-18:
        return float("nan")
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)


def spearman(scores: list[float], targets: list[float]) -> float:
    if len(set(scores)) < 2 or len(set(targets)) < 2:
        return float("nan")
    return pearson(rank_values(scores), rank_values(targets))


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if is_finite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def std_sample(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def with_fold_metadata(rows: list[dict[str, str]]) -> list[dict]:
    dates = sorted({row["date"] for row in rows})
    date_meta = {}
    for idx, date in enumerate(dates):
        date_meta[date] = {
            "fold": idx // FOLD_TEST_DAYS,
            "day_in_fold": idx % FOLD_TEST_DAYS,
        }
    out = []
    for row in rows:
        item = dict(row)
        item.update(date_meta[row["date"]])
        out.append(item)
    return out


def filter_sample(rows: list[dict], sample_name: str) -> list[dict]:
    if sample_name == "full_oos":
        return rows
    if sample_name == "purged_boundary_oos":
        return [row for row in rows if int(row["day_in_fold"]) >= HORIZON]
    raise ValueError(f"unknown sample: {sample_name}")


def grouped_by_date(rows: list[dict], score_col: str) -> dict[str, list[tuple[str, float, float]]]:
    groups: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for row in rows:
        score = to_float(row[score_col])
        target = to_float(row["future_return"])
        if is_finite(score) and is_finite(target):
            groups[row["date"]].append((row["asset"], score, target))
    return groups


def per_date_rank_corr(rows: list[dict], score_col: str) -> dict[str, float]:
    series = {}
    for date, values in grouped_by_date(rows, score_col).items():
        if len(values) < 2:
            continue
        scores = [v[1] for v in values]
        targets = [v[2] for v in values]
        rho = spearman(scores, targets)
        if is_finite(rho):
            series[date] = rho
    return dict(sorted(series.items()))


def top_k_backtest(rows: list[dict], score_col: str) -> dict[str, float]:
    groups = grouped_by_date(rows, score_col)
    dates = sorted(groups)
    rebal_dates = dates[::HORIZON]
    returns = []
    spreads = []
    turnovers = []
    prev_assets: set[str] | None = None

    for date in rebal_dates:
        values = sorted(groups[date], key=lambda item: item[1], reverse=True)
        if len(values) < TOP_K:
            continue
        top = values[:TOP_K]
        bottom = values[-TOP_K:]
        selected = {item[0] for item in top}
        if prev_assets is not None:
            turnovers.append(1.0 - len(selected & prev_assets) / float(TOP_K))
        prev_assets = selected
        top_ret = sum(item[2] for item in top) / TOP_K
        bottom_ret = sum(item[2] for item in bottom) / TOP_K
        returns.append(top_ret)
        spreads.append(top_ret - bottom_ret)

    ret_std = std_sample(returns)
    sharpe = (
        (sum(returns) / len(returns)) / ret_std * math.sqrt(252.0 / HORIZON)
        if returns and is_finite(ret_std) and ret_std > 1e-12
        else float("nan")
    )
    cumulative = 1.0
    for value in returns:
        cumulative *= 1.0 + value
    return {
        "top_k_cumulative_return": cumulative - 1.0 if returns else float("nan"),
        "top_k_sharpe": sharpe,
        "top_k_hit_rate": sum(1 for value in returns if value > 0.0) / len(returns) if returns else float("nan"),
        "top_bottom_spread_mean": sum(spreads) / len(spreads) if spreads else float("nan"),
        "turnover": sum(turnovers) / len(turnovers) if turnovers else 0.0,
        "n_rebalances": len(returns),
    }


def evaluate_candidate(rows: list[dict], sample_name: str, candidate: str, score_col: str) -> dict:
    sample = filter_sample(rows, sample_name)
    rank_series = per_date_rank_corr(sample, score_col)
    metrics = top_k_backtest(sample, score_col)
    dates = sorted({row["date"] for row in sample})
    return {
        "sample": sample_name,
        "candidate": candidate,
        "score_column": score_col,
        "n_rows": len(sample),
        "n_dates": len(dates),
        "start": dates[0] if dates else "",
        "end": dates[-1] if dates else "",
        "rank_corr": mean(rank_series.values()),
        **metrics,
    }


def stationary_block_sample(values: list[float], rng: random.Random, block_length: int) -> list[float]:
    if not values:
        return []
    p_new_block = 1.0 / block_length
    idx = rng.randrange(len(values))
    out = []
    while len(out) < len(values):
        out.append(values[idx])
        if rng.random() < p_new_block:
            idx = rng.randrange(len(values))
        else:
            idx = (idx + 1) % len(values)
    return out


def block_bootstrap_lift(
    rows: list[dict],
    sample_name: str,
    base_col: str,
    candidate_col: str,
) -> dict[str, float]:
    sample = filter_sample(rows, sample_name)
    base_series = per_date_rank_corr(sample, base_col)
    candidate_series = per_date_rank_corr(sample, candidate_col)
    common_dates = sorted(set(base_series) & set(candidate_series))
    diffs = [candidate_series[date] - base_series[date] for date in common_dates]
    observed = mean(diffs)
    rng = random.Random(SEED)
    draws = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample_diffs = stationary_block_sample(diffs, rng, BOOTSTRAP_BLOCK_LENGTH)
        draws.append(mean(sample_diffs))
    draws.sort()
    return {
        "rank_corr_lift": observed,
        "block_ci_low": percentile(draws, 0.025),
        "block_ci_high": percentile(draws, 0.975),
        "n_aligned_dates": len(common_dates),
    }


def build_candidate_metrics(rows: list[dict]) -> list[dict]:
    out = []
    for sample_name in ["full_oos", "purged_boundary_oos"]:
        for candidate, score_col in CANDIDATES.items():
            out.append(evaluate_candidate(rows, sample_name, candidate, score_col))
    return out


def build_bootstrap_rows(rows: list[dict]) -> list[dict]:
    out = []
    for sample_name in ["full_oos", "purged_boundary_oos"]:
        for label, base, candidate in LOCKED_COMPARISONS:
            boot = block_bootstrap_lift(
                rows,
                sample_name=sample_name,
                base_col=CANDIDATES[base],
                candidate_col=CANDIDATES[candidate],
            )
            out.append({
                "sample": sample_name,
                "comparison": label,
                "base": base,
                "candidate": candidate,
                **boot,
                "block_length": BOOTSTRAP_BLOCK_LENGTH,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
            })
    return out


def build_calibration_rows(calibrated_rows: list[dict[str, str]]) -> list[dict]:
    rows = with_fold_metadata(calibrated_rows)
    rows = [row for row in rows if is_finite(to_float(row.get("mu_calibrated_daily", "")))]
    out = []
    for sample_name in ["full_oos", "purged_boundary_oos"]:
        for label, score_col in [
            ("selected_mu_signal", "mu_signal"),
            ("selected_mu_calibrated_daily", "mu_calibrated_daily"),
        ]:
            out.append(evaluate_candidate(rows, sample_name, label, score_col))
    return out


def build_fold_boundary_audit(rows: list[dict]) -> list[dict]:
    dates = sorted({row["date"] for row in rows})
    out = []
    for fold_idx, start in enumerate(range(0, len(dates), FOLD_TEST_DAYS)):
        fold_dates = dates[start : start + FOLD_TEST_DAYS]
        if len(fold_dates) < FOLD_TEST_DAYS:
            continue
        dropped = fold_dates[:HORIZON]
        kept = fold_dates[HORIZON:]
        out.append(
            {
                "fold": fold_idx,
                "fold_start": fold_dates[0],
                "purged_start": dropped[0],
                "purged_end": dropped[-1],
                "kept_start": kept[0],
                "fold_end": fold_dates[-1],
                "raw_test_dates": len(fold_dates),
                "purged_dates": len(dropped),
                "kept_dates": len(kept),
                "purge_rule": f"drop first {HORIZON} dates of each {FOLD_TEST_DAYS}-date OOS fold",
            }
        )
    return out


def fmt(value: float) -> str:
    if not isinstance(value, float):
        return str(value)
    if not is_finite(value):
        return "nan"
    return f"{value:.4f}"


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def build_report(
    candidate_rows: list[dict],
    bootstrap_rows: list[dict],
    calibration_rows: list[dict],
    fold_audit_rows: list[dict],
    manifest: dict,
) -> str:
    purged = [row for row in candidate_rows if row["sample"] == "purged_boundary_oos"]
    full = [row for row in candidate_rows if row["sample"] == "full_oos"]
    purged_boot = [row for row in bootstrap_rows if row["sample"] == "purged_boundary_oos"]

    rank_candidate = next(row for row in purged if row["candidate"] == "mu_image_factor_rank")
    strict_candidate = next(row for row in purged if row["candidate"] == "mu_image_factor_strict_rank")
    sharpe_candidate = max(purged, key=lambda row: row["top_k_sharpe"])
    smoke = manifest.get("purged_retraining_smoke")
    smoke_lines = []
    if isinstance(smoke, dict) and smoke.get("status") == "PASS":
        smoke_lines = [
            "",
            "## Purged Retraining Smoke Test",
            "",
            "purged/embargoed 재학습 코드 경로가 실제로 동작하는지 1 fold / 1 epoch smoke test를 수행했다.",
            "",
            f"- 상태: `{smoke['status']}`",
            f"- 출력 폴더: `{smoke['output_dir']}`",
            f"- OOS rows: `{smoke['rows']}`",
            f"- OOS dates: `{smoke['dates']}`",
            f"- folds: `{smoke['folds']}`",
            f"- wf_embargo_days: `{smoke['wf_embargo_days']}`",
            f"- cnn_epochs: `{smoke['cnn_epochs']}`",
            f"- training history 생성: `{smoke['history_file_exists']}`",
            "",
            "이 smoke test는 성능 결론용이 아니라, purged fold 재학습 코드와 loss-history 기록 경로가 정상 동작함을 확인하는 용도다.",
        ]

    lines = [
        "# ODE `mu(t)` 논문 제출 보완 검증 작업공간",
        "",
        "## 목적",
        "",
        "이 작업공간은 `mu(t)` 입력 감사에서 남은 논문 제출 리스크를 보완하기 위한 별도 검증 결과다.",
        "CNN/LSTM 모델을 새로 재학습하지 않고, 이미 export된 OOS `mu(t)` handoff 파일을 대상으로",
        "fold-boundary purge와 후보 고정 평가를 수행한다.",
        "",
        "## 작업공간 설정",
        "",
        f"- 원본 파일: `{FINAL_WIDE.relative_to(ROOT)}`",
        f"- 출력 폴더: `{OUT_DIR.relative_to(ROOT)}`",
        f"- handoff grid 기준 test fold 길이: `{FOLD_TEST_DAYS}` dates",
        f"- purge 규칙: 각 fold의 첫 `{HORIZON}` OOS dates 제거",
        f"- Top-k: `{TOP_K}`",
        f"- Stationary block bootstrap: B=`{BOOTSTRAP_SAMPLES}`, mean block length=`{BOOTSTRAP_BLOCK_LENGTH}`",
        "",
        "## 핵심 결과",
        "",
        f"- Purged `mu_image_factor_rank` rank corr: `{rank_candidate['rank_corr']:.4f}`.",
        f"- Purged `mu_image_factor_strict_rank` rank corr: `{strict_candidate['rank_corr']:.4f}`.",
        f"- Purged Sharpe 최고 후보: `{sharpe_candidate['candidate']}`, Sharpe `{sharpe_candidate['top_k_sharpe']:.4f}`.",
        "",
        "해석: horizon-overlap 우려가 가장 큰 fold 시작부 날짜를 제거해도 주요 `mu(t)` 후보의 방향성은 유지된다.",
        "다만 rank-corr lift의 block-bootstrap CI는 0을 포함하므로, 통계적으로 확정된 alpha라고 표현하면 안 된다.",
        "",
        "## Fold-Boundary Purge Audit",
        "",
        markdown_table(
            fold_audit_rows[:5] + fold_audit_rows[-2:],
            [
                "fold",
                "fold_start",
                "purged_start",
                "purged_end",
                "kept_start",
                "fold_end",
                "raw_test_dates",
                "purged_dates",
                "kept_dates",
            ],
        ),
        "",
        f"총 `{len(fold_audit_rows)}`개 OOS fold에 대해 동일한 purge 규칙을 적용했다. 표는 앞 5개와 마지막 2개 fold만 보여준다.",
        "",
        "## 후보 고정 성능",
        "",
        markdown_table(
            full + purged,
            [
                "sample",
                "candidate",
                "n_dates",
                "rank_corr",
                "top_k_sharpe",
                "top_k_cumulative_return",
                "top_k_hit_rate",
                "turnover",
            ],
        ),
        "",
        "## Block Bootstrap Lift",
        "",
        markdown_table(
            bootstrap_rows,
            [
                "sample",
                "comparison",
                "rank_corr_lift",
                "block_ci_low",
                "block_ci_high",
                "n_aligned_dates",
            ],
        ),
        "",
        "## Calibration 점검",
        "",
        markdown_table(
            calibration_rows,
            [
                "sample",
                "candidate",
                "n_dates",
                "rank_corr",
                "top_k_sharpe",
                "top_k_hit_rate",
            ],
        ),
        "",
        "## 이번 보완으로 해결한 것",
        "",
        "- 별도 재현 가능 검증 작업공간을 추가했다.",
        "- 새 조합 탐색이 아니라 이미 고정한 최종 후보만 평가했다.",
        "- 20일 overlapping target 이슈에 대해 fold-boundary purged sensitivity를 추가했다.",
        "- 최종 lift 해석에 IID CI가 아니라 stationary block bootstrap CI를 사용했다.",
        "- embargo가 적용된 non-null calibration 구간에서 return-scale `mu`를 점검했다.",
        "",
        "## 남은 한계",
        "",
        "이 결과는 이미 학습된 OOS signal에 대한 sensitivity analysis다.",
        "따라서 purged/embargoed CNN 재학습을 완전히 대체하지는 못한다.",
        "더 엄격한 학회/저널 제출을 목표로 한다면, 학습 전에 label-overlap row를 제거하는 fold 구성으로 deep-learning stack을 다시 돌리는 것이 최종 보완이다.",
        "",
        "현재 코드에는 이 재학습을 위한 옵션이 추가되어 있다.",
        "",
        "```bash",
        "python build_image_factor_extension.py \\",
        "  --output-dir outputs/ode_handoff/06_mu_submission_validation/purged_retraining_candidate \\",
        "  --lookback 60 --horizon 20 \\",
        "  --wf-embargo-days 20 \\",
        "  --cnn-epochs 30 --patience 5 --weight-decay 5e-4",
        "```",
        "",
        "재학습 시 `cnn_training_history.csv`가 생성되어 fold별 train/validation loss와 best epoch를 확인할 수 있다.",
        *smoke_lines,
        "",
        "## 출력 파일",
        "",
        "- `candidate_locked_metrics.csv`",
        "- `purged_lift_block_bootstrap.csv`",
        "- `calibration_sanity_metrics.csv`",
        "- `fold_boundary_audit.csv`",
        "- `manifest.json`",
    ]
    _ = manifest
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = with_fold_metadata(read_csv(FINAL_WIDE))
    candidate_rows = build_candidate_metrics(rows)
    bootstrap_rows = build_bootstrap_rows(rows)
    calibration_rows = build_calibration_rows(read_csv(SELECTED_CALIBRATED))
    fold_audit_rows = build_fold_boundary_audit(rows)

    manifest = {
        "workspace": str(OUT_DIR.relative_to(ROOT)),
        "source_final_wide": str(FINAL_WIDE.relative_to(ROOT)),
        "source_selected_calibrated": str(SELECTED_CALIBRATED.relative_to(ROOT)),
        "horizon": HORIZON,
        "fold_test_days": FOLD_TEST_DAYS,
        "purge_rule": f"drop first {HORIZON} dates of each {FOLD_TEST_DAYS}-date OOS fold",
        "top_k": TOP_K,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_block_length": BOOTSTRAP_BLOCK_LENGTH,
        "seed": SEED,
        "candidate_count": len(CANDIDATES),
        "locked_comparisons": LOCKED_COMPARISONS,
        "purged_retraining_command": [
            "python",
            "build_image_factor_extension.py",
            "--output-dir",
            "outputs/ode_handoff/06_mu_submission_validation/purged_retraining_candidate",
            "--lookback",
            "60",
            "--horizon",
            "20",
            "--wf-embargo-days",
            "20",
            "--cnn-epochs",
            "30",
            "--patience",
            "5",
            "--weight-decay",
            "5e-4",
        ],
    }
    if SMOKE_SUMMARY.exists():
        manifest["purged_retraining_smoke"] = json.loads(SMOKE_SUMMARY.read_text(encoding="utf-8"))

    write_csv(OUT_DIR / "candidate_locked_metrics.csv", candidate_rows)
    write_csv(OUT_DIR / "purged_lift_block_bootstrap.csv", bootstrap_rows)
    write_csv(OUT_DIR / "calibration_sanity_metrics.csv", calibration_rows)
    write_csv(OUT_DIR / "fold_boundary_audit.csv", fold_audit_rows)
    write_json(OUT_DIR / "manifest.json", manifest)
    (OUT_DIR / "mu_submission_validation_report.md").write_text(
        build_report(candidate_rows, bootstrap_rows, calibration_rows, fold_audit_rows, manifest),
        encoding="utf-8",
    )

    print(json.dumps({
        "status": "PASS",
        "workspace": str(OUT_DIR.relative_to(ROOT)),
        "files": [
            "candidate_locked_metrics.csv",
            "purged_lift_block_bootstrap.csv",
            "calibration_sanity_metrics.csv",
            "fold_boundary_audit.csv",
            "manifest.json",
            "mu_submission_validation_report.md",
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
