"""Boosted SAR-transfer model with explicit assay-A uncertainty and pair history.

Only structure, assay-A measurements and training-partition labels are inputs.
The latter define a leave-one-out training pair prior and a full-training
validation/test prior. Hyperparameter variants are selected on validation AP.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from prepare_sar_transfer_20261007 import OUT, SEEDS, digest
from train_sar_transfer_baseline_20261007 import read, make_molecule_features, assay_features
from train_anchored_sar_transfer_20261007 import encode, metrics


RESULTS = OUT / "models_contrast_boost_20261007"
RESULTS.mkdir(exist_ok=True)


def pair_history(train: pd.DataFrame, frame: pd.DataFrame, *, loo: bool) -> np.ndarray:
    global_rate = float(train.binary_loss.mean())
    stats = train.groupby("pair_id").binary_loss.agg(["sum", "count"])
    successes = frame.pair_id.map(stats["sum"]).fillna(0).to_numpy(dtype=np.float32)
    counts = frame.pair_id.map(stats["count"]).fillna(0).to_numpy(dtype=np.float32)
    if loo:
        assert len(frame) == len(train)
        successes -= frame.binary_loss.to_numpy(dtype=np.float32)
        counts -= 1
    smoothed = (successes + 20 * global_rate) / (counts + 20)
    return np.column_stack((smoothed, np.log1p(counts))).astype(np.float32)


def a_uncertainty(frame: pd.DataFrame) -> np.ndarray:
    positive = frame.gap_A.to_numpy() > 0
    p1 = frame.p_A_1.to_numpy(dtype=np.float32)
    p2 = frame.p_A_2.to_numpy(dtype=np.float32)
    w1 = (frame.max_A_1 - frame.min_A_1).to_numpy(dtype=np.float32)
    w2 = (frame.max_A_2 - frame.min_A_2).to_numpy(dtype=np.float32)
    strong = np.where(positive, p1, p2)
    weak = np.where(positive, p2, p1)
    strong_width = np.where(positive, w1, w2)
    weak_width = np.where(positive, w2, w1)
    return np.column_stack((strong, weak, frame.robust_gap_A.to_numpy(dtype=np.float32),
                            strong_width, weak_width, strong_width + weak_width,
                            frame.same_scaffold.to_numpy(dtype=np.float32))).astype(np.float32)


def features(frame, blocks, history, variant, include_a_uncertainty=True):
    items = [blocks["molecule"], blocks["assay"]]
    if include_a_uncertainty:
        items.append(a_uncertainty(frame))
    if variant == "pair_aware":
        items.append(history)
    return np.concatenate(items, axis=1).astype(np.float32)


def main(seeds, include_a_uncertainty=True):
    results_dir = (RESULTS if include_a_uncertainty else
                   OUT / "models_contrast_boost_no_a_uncertainty_20261007")
    results_dir.mkdir(exist_ok=True)
    data = read("anchor_transfer_all.parquet")
    molecules = make_molecule_features(set(data.identity_1) | set(data.identity_2))
    rows = []
    for seed in seeds:
        split = read(f"anchor_split_scaffold_seed_{seed}.parquet")
        parts = {part: data.merge(split.loc[split.partition.eq(part), ["comparison_id"]],
                                  on="comparison_id", validate="one_to_one")
                 for part in ["train", "validation", "test"]}
        train = parts["train"]
        train_ids = set(train.identity_1) | set(train.identity_2)
        for part in ["validation", "test"]:
            assert train_ids.isdisjoint(set(parts[part].identity_1) | set(parts[part].identity_2))
        assays, numeric_median, tfidf, svd, formats = assay_features(train, pd.concat(parts.values()))
        targets = {x: np.zeros(1, dtype=np.float32) for x in data.target_chembl_id.unique()}
        blocks = {part: encode(frame, molecules, assays, numeric_median, targets)
                  for part, frame in parts.items()}
        history = {part: pair_history(train, frame, loo=(part == "train"))
                   for part, frame in parts.items()}
        y = train.binary_loss.to_numpy(dtype=np.int8)
        candidates = []
        for variant in ["pair_blind", "pair_aware"]:
            for depth in [3, 5]:
                xtr = features(train, blocks["train"], history["train"], variant, include_a_uncertainty)
                xval = features(parts["validation"], blocks["validation"], history["validation"], variant, include_a_uncertainty)
                model = XGBClassifier(
                    n_estimators=600, max_depth=depth, learning_rate=0.04,
                    min_child_weight=20, subsample=0.85, colsample_bytree=0.85,
                    reg_lambda=10, tree_method="hist", n_jobs=8,
                    objective="binary:logistic", eval_metric="logloss",
                    early_stopping_rounds=35, random_state=seed,
                )
                model.fit(xtr, y, eval_set=[(xval, parts["validation"].binary_loss)], verbose=False)
                pred_val = model.predict_proba(xval)[:, 1]
                met_val = metrics(parts["validation"], pred_val)
                candidates.append((met_val["AP"], variant, depth, model, met_val))
                rows.append({"seed": seed, "partition": "validation", "model": f"{variant}_d{depth}",
                             **met_val})
                print(seed, variant, depth, "best", model.best_iteration,
                      "val AP", round(met_val["AP"], 4), flush=True)
        _, variant, depth, best, _ = max(candidates, key=lambda row: row[0])
        test = parts["test"]
        pred = best.predict_proba(features(test, blocks["test"], history["test"], variant, include_a_uncertainty))[:, 1]
        test_metrics = metrics(test, pred)
        rows.append({"seed": seed, "partition": "test", "model": "validation_selected",
                     "selected_variant": variant, "selected_depth": depth, **test_metrics})
        pd.DataFrame(rows).to_csv(results_dir / "metrics.csv", index=False)
        output = test[["comparison_id", "pair_id", "target_chembl_id", "binary_loss", "outcome",
                       "source_type", "same_description"]].copy()
        output["prediction"] = pred
        output.to_csv(results_dir / f"predictions_seed_{seed}.csv", index=False)
        joblib.dump(best, results_dir / f"model_seed_{seed}.joblib", compress=3)
        joblib.dump({"tfidf": tfidf, "assay_svd": svd, "assay_formats": formats,
                     "numeric_median": numeric_median, "variant": variant, "depth": depth},
                    results_dir / f"preprocessors_seed_{seed}.joblib", compress=3)
        print(seed, "selected", variant, depth, "test", test_metrics, flush=True)
    (results_dir / "manifest.json").write_text(json.dumps({
        "method": "XGBoost molecular contrast, assay description, assay-A uncertainty, optional assay-pair prior",
        "model_selection": "Choose pair-prior inclusion and depth by validation AP; early stop on validation log-loss",
        "train_pair_prior": "Leave-one-comparison-out smoothed prior; heldout prior from training labels only",
        "include_a_uncertainty": include_a_uncertainty,
        "source_sha256": digest(OUT / "anchor_transfer_all.parquet"),
        "code_sha256": digest(Path(__file__)), "seeds": seeds,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--ablate-a-uncertainty", action="store_true")
    args = parser.parse_args()
    main(args.seeds, not args.ablate_a_uncertainty)
