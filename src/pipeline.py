"""End-to-end: weekly panel -> features -> CV -> hold-out -> forecast.

Run: python -m src.pipeline
(GDELT raw data must already be downloaded: python -m src.data.download_gdelt_bulk)
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
from src.data.build_weekly_panel import build_weekly_panel
from src.features.feature_engineering import FEATURE_COLUMNS, build_features, get_feature_columns
from src.models import baseline, ml_model, stat_model
from src.utils import load_config, repo_path, write_json
from src.validation.metrics import classification_report_full, regression_report
from src.validation.walk_forward import generate_folds, split_holdout

warnings.filterwarnings("ignore")

REG_MODELS = ["baseline_naive", "baseline_rollingmean", "ets", "xgboost"]
CLF_MODELS = ["baseline_naive", "xgboost"]


def _ets_forecast_all_horizons(train_df: pd.DataFrame, n_test: int, max_horizon: int) -> np.ndarray:
    return stat_model.fit_and_forecast(train_df["conflict_count"], steps=n_test + max_horizon)


def run_cv(cv_pool, folds, horizons, seed, feature_cols=None):
    feature_cols = feature_cols or FEATURE_COLUMNS
    results = {h: {m: {"y_true": [], "y_pred": []} for m in REG_MODELS} for h in horizons}
    results_event = {h: {m: {"y_true": [], "y_prob": []} for m in CLF_MODELS} for h in horizons}
    max_h = max(horizons)

    for fold_i, (train_range, test_range) in enumerate(folds):
        train_df = cv_pool.iloc[list(train_range)]
        test_df = cv_pool.iloc[list(test_range)]
        n_test = len(test_df)
        ets_fc = _ets_forecast_all_horizons(train_df, n_test, max_h)

        for h in horizons:
            tgt_col = f"target_count_h{h}"
            y_true = test_df[tgt_col].to_numpy(dtype=float)
            valid = ~np.isnan(y_true)
            preds = {
                "baseline_naive": baseline.predict_naive(test_df, h),
                "baseline_rollingmean": baseline.predict_rolling_mean(test_df, h),
                "ets": ets_fc[h - 1: h - 1 + n_test],
                "xgboost": ml_model.fit_predict_regressor(train_df, test_df, feature_cols, tgt_col, seed),
            }
            if valid.sum() == 0:
                continue
            for m in REG_MODELS:
                results[h][m]["y_true"].append(y_true[valid])
                results[h][m]["y_pred"].append(np.asarray(preds[m])[valid])

            evt_col = f"target_event_h{h}"
            evt_true = test_df[evt_col]
            valid = evt_true.notna()
            if valid.sum() == 0:
                continue
            y_evt = evt_true[valid].to_numpy(dtype=int)
            evt_preds = {
                "baseline_naive": baseline.predict_escalation_naive(test_df, h)[valid.to_numpy()],
                "xgboost": ml_model.fit_predict_classifier(train_df, test_df, feature_cols, evt_col, seed)[valid.to_numpy()],
            }
            for m in CLF_MODELS:
                results_event[h][m]["y_true"].append(y_evt)
                results_event[h][m]["y_prob"].append(evt_preds[m])

        print(f"[cv] fold {fold_i}: train={len(train_df)} test={n_test} "
              f"({test_df['week_start'].min()} .. {test_df['week_start'].max()})")
    return results, results_event


def summarize_cv(results, results_event, train_history) -> dict:
    summary = {"regression": {}, "classification": {}}
    for h, models in results.items():
        y_naive = np.concatenate(models["baseline_naive"]["y_pred"])
        summary["regression"][h] = {
            m: regression_report(np.concatenate(d["y_true"]), np.concatenate(d["y_pred"]), y_naive)
            for m, d in models.items()
        }
    for h, models in results_event.items():
        summary["classification"][h] = {}
        for m, d in models.items():
            if d["y_true"]:
                summary["classification"][h][m] = classification_report_full(
                    np.concatenate(d["y_true"]), np.concatenate(d["y_prob"])
                )
    return summary


def run_holdout(cv_pool, full_df, holdout_target_weeks, horizons, seed, feature_cols=None):
    feature_cols = feature_cols or FEATURE_COLUMNS
    max_h = max(horizons)
    last_week = pd.to_datetime(full_df["week_start"]).max()
    cv_last_week = pd.to_datetime(cv_pool["week_start"]).max()
    ets_steps = int((last_week - cv_last_week) / pd.Timedelta(weeks=1)) + max_h
    ets_fc = stat_model.fit_and_forecast(cv_pool["conflict_count"], steps=ets_steps)

    out = {"regression": {}, "classification": {}}
    for h in horizons:
        origins = {w - dt.timedelta(weeks=h) for w in holdout_target_weeks}
        test_df = full_df[full_df["week_start"].isin(origins)].sort_values("week_start").reset_index(drop=True)

        tgt_col = f"target_count_h{h}"
        y_true = test_df[tgt_col].to_numpy(dtype=float)
        valid = ~np.isnan(y_true)
        steps_ahead = (
            (pd.to_datetime(test_df["week_start"]) + pd.Timedelta(weeks=h) - cv_last_week)
            / pd.Timedelta(weeks=1)
        ).astype(int)
        preds = {
            "baseline_naive": baseline.predict_naive(test_df, h),
            "baseline_rollingmean": baseline.predict_rolling_mean(test_df, h),
            "ets": ets_fc[steps_ahead.to_numpy() - 1],
            "xgboost": ml_model.fit_predict_regressor(cv_pool, test_df, feature_cols, tgt_col, seed),
        }
        week_labels = [str(w + dt.timedelta(weeks=h)) for w, v in zip(test_df["week_start"], valid) if v]
        y_hat = {m: np.asarray(preds[m])[valid] for m in REG_MODELS}
        y_naive = y_hat["baseline_naive"]
        out["regression"][h] = {
            m: {
                **regression_report(y_true[valid], y_hat[m], y_naive),
                "y_true": y_true[valid].tolist(),
                "y_pred": y_hat[m].tolist(),
                "week_start": week_labels,
            }
            for m in REG_MODELS
        }

        evt_col = f"target_event_h{h}"
        evt_true = test_df[evt_col]
        valid = evt_true.notna().to_numpy()
        if valid.sum() > 0:
            y_evt = evt_true[valid].to_numpy(dtype=int)
            evt_preds = {
                "baseline_naive": baseline.predict_escalation_naive(test_df, h)[valid],
                "xgboost": ml_model.fit_predict_classifier(cv_pool, test_df, feature_cols, evt_col, seed)[valid],
            }
            out["classification"][h] = {
                m: classification_report_full(y_evt, evt_preds[m]) for m in CLF_MODELS
            }
    return out


def compute_mase_vs_baseline_table(holdout_summary) -> pd.DataFrame:
    rows = []
    for h, models in holdout_summary["regression"].items():
        base = models["baseline_naive"]["MASE"]
        for m, rep in models.items():
            vs = None if m == "baseline_naive" or not base else (base - rep["MASE"]) / base * 100
            rows.append({
                "horizon": h, "model": m,
                "MASE": round(rep["MASE"], 3),
                "sMAPE": round(rep["sMAPE"], 2),
                "RMSE": round(rep["RMSE"], 2),
                "vs_baseline_pct": None if vs is None else round(vs, 1),
            })
    return pd.DataFrame(rows)


def make_figures(feature_df, holdout_summary, outputs_dir: Path, country: str, code: str):
    figs_dir = outputs_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(pd.to_datetime(feature_df["week_start"]), feature_df["conflict_count"], color="steelblue", lw=1)
    spikes = feature_df[feature_df["escalation_event"] == 1]
    ax.scatter(pd.to_datetime(spikes["week_start"]), spikes["conflict_count"],
               color="crimson", s=14, label="escalation_event=1", zorder=3)
    ax.set_title(f"{country} ({code}) — weekly conflict events (CAMEO 18/19)")
    ax.set_ylabel("conflict_count / week")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "weekly_conflict_count_history.png", dpi=130)
    plt.close(fig)

    reg = holdout_summary["regression"][1]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    weeks = pd.to_datetime(reg["xgboost"]["week_start"])
    ax.plot(weeks, reg["xgboost"]["y_true"], "o-", color="black", label="actual", lw=2)
    for m, style in [("baseline_naive", "--"), ("ets", ":"), ("xgboost", "-")]:
        ax.plot(weeks, reg[m]["y_pred"], style, label=m)
    ax.set_title("Hold-out, horizon=1 week")
    ax.set_ylabel("conflict_count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "holdout_actual_vs_predicted_h1.png", dpi=130)
    plt.close(fig)

    table = compute_mase_vs_baseline_table(holdout_summary)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in table["model"].unique():
        sub = table[table["model"] == m]
        ax.plot(sub["horizon"], sub["MASE"], marker="o", label=m)
    ax.axhline(1.0, color="grey", lw=1, ls="--")
    ax.set_xlabel("horizon (weeks)")
    ax.set_ylabel("MASE (hold-out)")
    ax.set_title("Hold-out MASE by model and horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "mase_by_horizon.png", dpi=130)
    plt.close(fig)
    print(f"[pipeline] figures -> {figs_dir}")


def build_future_forecast(feature_df, config, seed, feature_cols=None) -> pd.DataFrame:
    feature_cols = feature_cols or FEATURE_COLUMNS
    country_name = config["country"]["name"]
    last_row = feature_df.iloc[[-1]]
    last_week = pd.to_datetime(last_row["week_start"].iloc[0])
    rows = []

    for h in config["target"]["horizons"]:
        tgt_col = f"target_count_h{h}"
        train = feature_df.dropna(subset=[tgt_col])
        model = ml_model.fit_full_regressor(train, feature_cols, tgt_col, seed)
        pred = float(max(model.predict(last_row[feature_cols])[0], 0.0))
        resid = train[tgt_col] - model.predict(train[feature_cols])
        sigma = float(np.std(resid))
        ci_lo, ci_hi = max(pred - 1.645 * sigma, 0.0), pred + 1.645 * sigma

        forecast_week = last_week + pd.Timedelta(weeks=h)
        iso_year, iso_wk, _ = forecast_week.isocalendar()
        rows.append({
            "country": country_name, "forecast_week": f"{iso_year}-W{iso_wk:02d}", "horizon": h,
            "target_type": "count", "predicted_value": round(pred, 1), "predicted_prob": "",
            "ci_lower": round(ci_lo, 1), "ci_upper": round(ci_hi, 1), "model": "xgboost",
        })

        evt_col = f"target_event_h{h}"
        train_evt = feature_df.dropna(subset=[evt_col])
        if train_evt[evt_col].nunique() < 2:
            prob = 0.0
        else:
            clf = ml_model.fit_full_classifier(train_evt, feature_cols, evt_col, seed)
            prob = float(clf.predict_proba(last_row[feature_cols])[0, 1])
        rows.append({
            "country": country_name, "forecast_week": f"{iso_year}-W{iso_wk:02d}", "horizon": h,
            "target_type": "event", "predicted_value": "", "predicted_prob": round(prob, 2),
            "ci_lower": "", "ci_upper": "", "model": "xgboost",
        })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    seed = cfg["random_seed"]
    horizons = cfg["target"]["horizons"]
    max_h = max(horizons)

    weekly = build_weekly_panel(cfg)
    feature_df = build_features(weekly, cfg, horizons)
    feature_cols = get_feature_columns(feature_df)

    cv_pool, full_df, holdout_start, holdout_target_weeks = split_holdout(
        feature_df, cfg["validation"]["holdout_weeks"], max_h
    )
    print(f"[pipeline] CV pool={len(cv_pool)} weeks, hold-out={len(holdout_target_weeks)} from {holdout_start}")

    folds = generate_folds(
        len(cv_pool),
        cfg["validation"]["cv_initial_train_weeks"],
        cfg["validation"]["cv_step_weeks"],
        cfg["validation"]["min_test_weeks_per_fold"],
    )
    print(f"[pipeline] {len(folds)} CV folds, {len(feature_cols)} features")

    cv_results, cv_results_event = run_cv(cv_pool, folds, horizons, seed, feature_cols=feature_cols)
    cv_summary = summarize_cv(cv_results, cv_results_event, cv_pool["conflict_count"].to_numpy())
    holdout_summary = run_holdout(cv_pool, full_df, holdout_target_weeks, horizons, seed, feature_cols=feature_cols)
    results_table = compute_mase_vs_baseline_table(holdout_summary)

    outputs_dir = repo_path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    make_figures(feature_df, holdout_summary, outputs_dir,
                 cfg["country"]["name"], cfg["country"]["gdelt_code"])

    future_forecast = build_future_forecast(feature_df, cfg, seed, feature_cols=feature_cols)
    future_forecast.to_csv(outputs_dir / "forecast.csv", index=False)
    print(future_forecast.to_string(index=False))

    metrics_out = {
        "config": cfg,
        "n_weeks_total": int(len(feature_df)),
        "cv_n_folds": len(folds),
        "cv_summary": cv_summary,
        "holdout_summary": {
            "regression": {
                h: {m: {k: v for k, v in rep.items() if k not in ("y_true", "y_pred", "week_start")}
                    for m, rep in models.items()}
                for h, models in holdout_summary["regression"].items()
            },
            "classification": holdout_summary["classification"],
        },
        "results_table_holdout": results_table.to_dict(orient="records"),
    }
    write_json(outputs_dir / "metrics.json", metrics_out)
    print("\n=== HOLD-OUT ===")
    print(results_table.to_string(index=False))


if __name__ == "__main__":
    main()
