"""
End-to-end orchestration: data -> features -> models -> walk-forward CV ->
hold-out evaluation -> future forecast -> outputs/{metrics.json, forecast.csv,
figures/}.

Run with:  python -m src.pipeline
(Assumes GDELT raw data has already been downloaded via
 `python -m src.data.download_gdelt_bulk` -- see README quick start.)
"""
from __future__ import annotations

import datetime as dt
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.build_weekly_panel import build_weekly_panel  # noqa: E402
from src.features.feature_engineering import FEATURE_COLUMNS, build_features  # noqa: E402
from src.models import baseline, ml_model, stat_model  # noqa: E402
from src.utils import load_config, repo_path, write_json  # noqa: E402
from src.validation.metrics import classification_report_full, regression_report  # noqa: E402
from src.validation.walk_forward import generate_folds, split_holdout  # noqa: E402

warnings.filterwarnings("ignore")


def _ets_forecast_all_horizons(train_df: pd.DataFrame, n_test: int, max_horizon: int) -> np.ndarray:
    """Fit ETS once per fold and return the full forecast array (steps =
    n_test + max_horizon); callers slice `fc[h-1 : h-1+n_test]` per horizon."""
    steps = n_test + max_horizon
    return stat_model.fit_and_forecast(train_df["conflict_count"], steps=steps)


def run_cv(cv_pool: pd.DataFrame, folds, horizons: list[int], seed: int) -> dict:
    """Returns nested dict: results[h][model_name] = {'y_true':..., 'y_pred':...}
    plus for classification: results_event[h][model_name] = {'y_true':..., 'y_prob':...}
    Out-of-fold predictions are concatenated across all folds."""
    reg_models = ["baseline_naive", "baseline_rollingmean", "ets", "xgboost"]
    clf_models = ["baseline_naive", "xgboost"]
    results = {h: {m: {"y_true": [], "y_pred": []} for m in reg_models} for h in horizons}
    results_event = {h: {m: {"y_true": [], "y_prob": []} for m in clf_models} for h in horizons}

    max_h = max(horizons)
    for fold_i, (train_range, test_range) in enumerate(folds):
        train_df = cv_pool.iloc[list(train_range)]
        test_df = cv_pool.iloc[list(test_range)]
        n_test = len(test_df)
        ets_fc = _ets_forecast_all_horizons(train_df, n_test, max_h)
        for h in horizons:
            tgt_col = f"target_count_h{h}"
            y_true = test_df[tgt_col].to_numpy(dtype=float)
            # data-quality mask (see feature_engineering.py): drop test rows
            # whose target week had incomplete GDELT coverage -- their
            # "true" count is a known undercount and must not be scored.
            valid = ~np.isnan(y_true)

            preds = {
                "baseline_naive": baseline.predict_naive(test_df, h),
                "baseline_rollingmean": baseline.predict_rolling_mean(test_df, h),
                "ets": ets_fc[h - 1: h - 1 + n_test],
                "xgboost": ml_model.fit_predict_regressor(train_df, test_df, FEATURE_COLUMNS, tgt_col, seed),
            }
            if valid.sum() == 0:
                continue
            for m in reg_models:
                results[h][m]["y_true"].append(y_true[valid])
                results[h][m]["y_pred"].append(np.asarray(preds[m])[valid])

            evt_col = f"target_event_h{h}"
            evt_true = test_df[evt_col]
            valid = evt_true.notna()
            if valid.sum() == 0:
                continue
            y_evt_true = evt_true[valid].to_numpy(dtype=int)
            evt_preds = {
                "baseline_naive": baseline.predict_escalation_naive(test_df, h)[valid.to_numpy()],
                "xgboost": ml_model.fit_predict_classifier(train_df, test_df, FEATURE_COLUMNS, evt_col, seed)[valid.to_numpy()],
            }
            for m in clf_models:
                results_event[h][m]["y_true"].append(y_evt_true)
                results_event[h][m]["y_prob"].append(evt_preds[m])
        print(f"[pipeline] CV fold {fold_i}: train={len(train_df)} rows, test={n_test} rows "
              f"({test_df['week_start'].min()} .. {test_df['week_start'].max()})")
    return results, results_event


def summarize_cv(results: dict, results_event: dict, train_history: np.ndarray) -> dict:
    summary = {"regression": {}, "classification": {}}
    for h, models in results.items():
        summary["regression"][h] = {}
        for m, d in models.items():
            y_true = np.concatenate(d["y_true"])
            y_pred = np.concatenate(d["y_pred"])
            summary["regression"][h][m] = regression_report(y_true, y_pred, train_history)
    for h, models in results_event.items():
        summary["classification"][h] = {}
        for m, d in models.items():
            if not d["y_true"]:
                continue
            y_true = np.concatenate(d["y_true"])
            y_prob = np.concatenate(d["y_prob"])
            summary["classification"][h][m] = classification_report_full(y_true, y_prob)
    return summary


def run_holdout(cv_pool: pd.DataFrame, full_df: pd.DataFrame, holdout_target_weeks: set, horizons: list[int], seed: int) -> dict:
    """Evaluate models -- FIT ONLY ON cv_pool -- against the last `holdout_weeks`
    realized weeks. For horizon h, the test set is every row of `full_df`
    whose target realization week (week_start + h weeks) falls inside
    `holdout_target_weeks`. See walk_forward.split_holdout for why test-row
    ORIGINS may legitimately fall inside the hold-out window while the MODEL
    itself never trains on hold-out data."""
    reg_models = ["baseline_naive", "baseline_rollingmean", "ets", "xgboost"]
    clf_models = ["baseline_naive", "xgboost"]
    max_h = max(horizons)
    last_week = pd.to_datetime(full_df["week_start"]).max()
    cv_last_week = pd.to_datetime(cv_pool["week_start"]).max()
    ets_steps = int((last_week - cv_last_week) / pd.Timedelta(weeks=1)) + max_h
    ets_fc = stat_model.fit_and_forecast(cv_pool["conflict_count"], steps=ets_steps)

    out = {"regression": {}, "classification": {}}
    for h in horizons:
        target_weeks_h = {w - dt.timedelta(weeks=h) for w in holdout_target_weeks}
        mask = full_df["week_start"].isin(target_weeks_h)
        test_df = full_df[mask].sort_values("week_start").reset_index(drop=True)

        tgt_col = f"target_count_h{h}"
        y_true = test_df[tgt_col].to_numpy(dtype=float)
        valid = ~np.isnan(y_true)  # data-quality mask, see run_cv
        # ETS: steps-ahead from cv_pool's last training week to each test row's target week
        steps_ahead = ((pd.to_datetime(test_df["week_start"]) + pd.Timedelta(weeks=h) - cv_last_week)
                       / pd.Timedelta(weeks=1)).astype(int)
        ets_preds = ets_fc[steps_ahead.to_numpy() - 1]
        preds = {
            "baseline_naive": baseline.predict_naive(test_df, h),
            "baseline_rollingmean": baseline.predict_rolling_mean(test_df, h),
            "ets": ets_preds,
            "xgboost": ml_model.fit_predict_regressor(cv_pool, test_df, FEATURE_COLUMNS, tgt_col, seed),
        }
        target_week_labels = [str(w + dt.timedelta(weeks=h)) for w, v in zip(test_df["week_start"], valid) if v]
        out["regression"][h] = {
            m: {**regression_report(y_true[valid], np.asarray(preds[m])[valid], cv_pool["conflict_count"].to_numpy()),
                "y_true": y_true[valid].tolist(), "y_pred": np.asarray(preds[m])[valid].tolist(),
                "week_start": target_week_labels}
            for m in reg_models
        }

        evt_col = f"target_event_h{h}"
        evt_true = test_df[evt_col]
        valid = evt_true.notna().to_numpy()
        if valid.sum() > 0:
            y_evt_true = evt_true[valid].to_numpy(dtype=int)
            evt_preds = {
                "baseline_naive": baseline.predict_escalation_naive(test_df, h)[valid],
                "xgboost": ml_model.fit_predict_classifier(cv_pool, test_df, FEATURE_COLUMNS, evt_col, seed)[valid],
            }
            out["classification"][h] = {
                m: classification_report_full(y_evt_true, evt_preds[m]) for m in clf_models
            }
    return out


def compute_mase_vs_baseline_table(holdout_summary: dict) -> pd.DataFrame:
    rows = []
    for h, models in holdout_summary["regression"].items():
        baseline_mase = models["baseline_naive"]["MASE"]
        for m, rep in models.items():
            improvement = None
            if m != "baseline_naive" and baseline_mase:
                improvement = (baseline_mase - rep["MASE"]) / baseline_mase * 100
            rows.append({
                "horizon": h, "model": m, "MASE": round(rep["MASE"], 3),
                "sMAPE": round(rep["sMAPE"], 2), "RMSE": round(rep["RMSE"], 2),
                "vs_baseline_pct": None if improvement is None else round(improvement, 1),
            })
    return pd.DataFrame(rows)


def make_figures(feature_df: pd.DataFrame, holdout_summary: dict, outputs_dir: Path):
    figs_dir = outputs_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full history of weekly conflict_count with escalation flags
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(pd.to_datetime(feature_df["week_start"]), feature_df["conflict_count"], color="steelblue", lw=1)
    spikes = feature_df[feature_df["escalation_event"] == 1]
    ax.scatter(pd.to_datetime(spikes["week_start"]), spikes["conflict_count"], color="crimson", s=14, label="escalation_event=1", zorder=3)
    ax.set_title("Ukraine (UP) - Weekly conflict events (CAMEO 18/19) & flagged escalations")
    ax.set_ylabel("conflict_count / week")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "weekly_conflict_count_history.png", dpi=130)
    plt.close(fig)

    # 2. Hold-out: actual vs predicted for horizon=1, all models
    h = 1
    reg = holdout_summary["regression"][h]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    weeks = pd.to_datetime(reg["xgboost"]["week_start"])
    ax.plot(weeks, reg["xgboost"]["y_true"], "o-", color="black", label="actual", lw=2)
    for m, style in [("baseline_naive", "--"), ("ets", ":"), ("xgboost", "-")]:
        ax.plot(weeks, reg[m]["y_pred"], style, label=m)
    ax.set_title(f"Hold-out (last 12 weeks): actual vs predicted, horizon={h} week")
    ax.set_ylabel("conflict_count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / f"holdout_actual_vs_predicted_h{h}.png", dpi=130)
    plt.close(fig)

    # 3. MASE by model / horizon bar chart
    table = compute_mase_vs_baseline_table(holdout_summary)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in table["model"].unique():
        sub = table[table["model"] == m]
        ax.plot(sub["horizon"], sub["MASE"], marker="o", label=m)
    ax.axhline(1.0, color="grey", lw=1, ls="--")
    ax.set_xlabel("horizon (weeks)")
    ax.set_ylabel("MASE (hold-out)")
    ax.set_title("Hold-out MASE by model and forecast horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "mase_by_horizon.png", dpi=130)
    plt.close(fig)

    print(f"[pipeline] Saved figures to {figs_dir}")


def build_future_forecast(feature_df: pd.DataFrame, config: dict, seed: int) -> pd.DataFrame:
    """Train final models on ALL rows with a defined target, then forecast
    the h-week-ahead count & escalation probability from the very last
    (most recently fully observed) origin week in the panel."""
    horizons = config["target"]["horizons"]
    country_name = config["country"]["name"]
    last_row = feature_df.iloc[[-1]]
    last_week = pd.to_datetime(last_row["week_start"].iloc[0])

    rows = []
    for h in horizons:
        tgt_col = f"target_count_h{h}"
        train = feature_df.dropna(subset=[tgt_col])
        model = ml_model.fit_full_regressor(train, FEATURE_COLUMNS, tgt_col, seed)
        pred = float(max(model.predict(last_row[FEATURE_COLUMNS])[0], 0.0))

        # 90% empirical prediction interval via XGBoost quantile regression
        try:
            import xgboost as xgb

            lo_model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.05,
                                         n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed)
            hi_model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.95,
                                         n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed)
            lo_model.fit(train[FEATURE_COLUMNS], train[tgt_col])
            hi_model.fit(train[FEATURE_COLUMNS], train[tgt_col])
            ci_lo = float(max(lo_model.predict(last_row[FEATURE_COLUMNS])[0], 0.0))
            ci_hi = float(max(hi_model.predict(last_row[FEATURE_COLUMNS])[0], ci_lo))
        except Exception:  # noqa: BLE001
            resid_std = float(np.std(train[tgt_col] - model.predict(train[FEATURE_COLUMNS])))
            ci_lo, ci_hi = max(pred - 1.645 * resid_std, 0.0), pred + 1.645 * resid_std

        forecast_week = last_week + pd.Timedelta(weeks=h)
        iso_year, iso_wk, _ = forecast_week.isocalendar()
        rows.append({
            "country": country_name, "forecast_week": f"{iso_year}-W{iso_wk:02d}", "horizon": h,
            "target_type": "count", "predicted_value": round(pred, 1), "predicted_prob": "",
            "ci_lower": round(ci_lo, 1), "ci_upper": round(ci_hi, 1), "model": "xgboost",
        })

        evt_col = f"target_event_h{h}"
        train_evt = feature_df.dropna(subset=[evt_col])
        clf = ml_model.fit_full_classifier(train_evt, FEATURE_COLUMNS, evt_col, seed)
        prob = float(clf.predict_proba(last_row[FEATURE_COLUMNS])[0, 1])
        # bagged ensemble for a rough probability CI
        rng = np.random.RandomState(seed)
        boot_probs = []
        for b in range(15):
            sample = train_evt.sample(frac=0.8, random_state=rng.randint(0, 1_000_000))
            if sample[evt_col].nunique() < 2:
                continue
            m_b = ml_model.fit_full_classifier(sample, FEATURE_COLUMNS, evt_col, seed + b)
            boot_probs.append(float(m_b.predict_proba(last_row[FEATURE_COLUMNS])[0, 1]))
        if boot_probs:
            p_lo, p_hi = np.percentile(boot_probs, [5, 95])
        else:
            p_lo, p_hi = prob, prob
        rows.append({
            "country": country_name, "forecast_week": f"{iso_year}-W{iso_wk:02d}", "horizon": h,
            "target_type": "event", "predicted_value": "", "predicted_prob": round(prob, 2),
            "ci_lower": round(float(p_lo), 2), "ci_upper": round(float(p_hi), 2), "model": "xgboost",
        })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    seed = cfg["random_seed"]
    horizons = cfg["target"]["horizons"]
    max_h = max(horizons)

    weekly = build_weekly_panel(cfg)
    feature_df = build_features(weekly, cfg, horizons)

    cv_pool, full_df, holdout_start, holdout_target_weeks = split_holdout(
        feature_df, cfg["validation"]["holdout_weeks"], max_h
    )
    print(f"[pipeline] CV pool: {len(cv_pool)} weeks | Hold-out: {len(holdout_target_weeks)} realized weeks "
          f"starting {holdout_start}")

    folds = generate_folds(
        len(cv_pool),
        cfg["validation"]["cv_initial_train_weeks"],
        cfg["validation"]["cv_step_weeks"],
        cfg["validation"]["min_test_weeks_per_fold"],
    )
    print(f"[pipeline] Generated {len(folds)} walk-forward CV folds")

    cv_results, cv_results_event = run_cv(cv_pool, folds, horizons, seed)
    cv_summary = summarize_cv(cv_results, cv_results_event, cv_pool["conflict_count"].to_numpy())

    holdout_summary = run_holdout(cv_pool, full_df, holdout_target_weeks, horizons, seed)
    results_table = compute_mase_vs_baseline_table(holdout_summary)

    outputs_dir = repo_path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    make_figures(feature_df, holdout_summary, outputs_dir)

    future_forecast = build_future_forecast(feature_df, cfg, seed)
    future_forecast.to_csv(outputs_dir / "forecast.csv", index=False)
    print(f"[pipeline] Wrote {outputs_dir / 'forecast.csv'}")
    print(future_forecast.to_string(index=False))

    metrics_out = {
        "config": cfg,
        "n_weeks_total": int(len(feature_df)),
        "cv_n_folds": len(folds),
        "cv_summary": cv_summary,
        "holdout_summary": {
            "regression": {h: {m: {k: v for k, v in rep.items() if k not in ("y_true", "y_pred", "week_start")}
                                for m, rep in models.items()}
                           for h, models in holdout_summary["regression"].items()},
            "classification": holdout_summary["classification"],
        },
        "results_table_holdout": results_table.to_dict(orient="records"),
    }
    write_json(outputs_dir / "metrics.json", metrics_out)
    print(f"[pipeline] Wrote {outputs_dir / 'metrics.json'}")
    print("\n=== HOLD-OUT RESULTS TABLE ===")
    print(results_table.to_string(index=False))


if __name__ == "__main__":
    main()
