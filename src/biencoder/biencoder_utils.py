import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    precision_recall_curve,
    roc_curve,
)


def flatten_dict(d, prefix=""):
    output = dict()
    for k, v in d.items():
        key = f"{prefix}{k}"
        # print(f"{prefix=}, {k=}, {v=}")
        if isinstance(v, dict):
            output.update(flatten_dict(v, prefix=f"{key}__"))
        else:
            output[key] = v
    # print(f"{prefix=}, {output=}")
    return output


def unflatten_dict(d):
    output = defaultdict(dict)
    for k, v in d.items():
        keys = k.split("__", 1)
        # print(k, keys, v)
        if len(keys) > 1:
            output[keys[0]][keys[1]] = v
        else:
            output[keys[0]] = v
    for k, v in output.items():
        if isinstance(v, dict):
            output[k] = unflatten_dict(v)
    return output


def print_eval_metrics(eval_dir, file_suffix=""):
    data = []
    eval_files = Path(eval_dir).glob("./*_results.json")
    # for split in ["train", "test"]:
    splits = []
    for eval_file in eval_files:
        # eval_file = f"{eval_dir}/{split}_results.json"
        split = eval_file.name.replace("_results.json", "")
        print(f"{split=}, {eval_file=}")
        if split in {"all", "train_result"}:
            continue
        splits.append(split)
        with open(eval_file) as fp:
            data.append(
                pd.Series(json.load(fp))
                .rename("value")
                .to_frame()
                .reset_index()
                .rename(columns={"index": "key"})
                .assign(split=split)
            )
            data[-1]["metric"] = data[-1]["key"].str.rsplit(
                f"{split}_", n=1, expand=True
            )[1]
    data = pd.concat(data)
    print(data.head())

    df_t = data.pipe(
        lambda df_t: pd.concat(
            [
                df_t,
                # df_t["key"].str.replace(df_t["split"], "_").str.rsplit("_", n=1, expand=True),
                df_t.metric.str.extract(r"layer_(?P<layer>[\d]+)_dim_(?P<dim>[\d]+)")
                .fillna({"layer": 12, "dim": 768})
                .astype(int),
                df_t.metric.str.replace(
                    r"layer_(?P<layer>[\d]+)_dim_(?P<dim>[\d]+)__", "", regex=True
                ).rename("metric_name"),
            ],
            axis=1,
        ).sort_values(["layer", "dim"])
    )
    # print metrics
    df_metrics = df_t[
        df_t.metric_name.isin(
            [
                "macro avg__f1-score",
                "macro avg__precision",
                "macro avg__recall",
                "accuracy",
            ]
        )
    ].pivot(index=["layer", "dim"], columns=["split", "metric_name"], values="value")
    print(df_metrics)
    df_metrics.to_csv(f"{eval_dir}/df_metrics{file_suffix}.csv", sep="\t")

    # Plot
    num_cols = 2
    num_rows = math.ceil(len(splits) / num_cols)

    fig, ax = plt.subplots(
        num_rows,
        num_cols,
        sharex=False,
        sharey=False,
        figsize=(8 * num_cols, 5 * num_rows),
    )
    for split, axi in zip(splits, ax.ravel()):
        df_metrics[split].plot(
            kind="line",
            marker="o",
            color=["red", "steelblue", "olive", "orange"],
            ax=axi,
        )
        axi.set_title(split)
    fig.tight_layout()

    plt.savefig(f"{eval_dir}/df_metrics{file_suffix}.png", bbox_inches="tight")


def make_prediction_analysis_plots(
    df, pred_cols, label_col="label_ids", split="test", label_threshold=0
):
    y_test = df[label_col] > label_threshold
    prevalence_pos_label = y_test.mean()
    print(f"{prevalence_pos_label=:.3f}, {split=}")
    fig, ax = plt.subplots(1, 5, figsize=(8 * 5, 6), squeeze=False)

    for i, pred_col in enumerate(pred_cols):
        print(pred_col)
        y_score = df[pred_col]

        prec, recall, thresholds = precision_recall_curve(y_test, y_score)
        pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot(
            ax=ax[0][0], label=pred_col
        )
        l = ax[0][1].plot(thresholds, prec[:-1], label=f"{pred_col} [P]")
        color = l[0].get_color()
        ax[0][1].plot(
            thresholds,
            recall[:-1],
            label=f"{pred_col} [R]",
            linestyle="--",
            color=color,
        )
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(
            ax=ax[0][2], label=pred_col, plot_chance_level=i == 0
        )

        prob_true, prob_pred = calibration_curve(y_test, y_score, n_bins=20)
        disp = CalibrationDisplay(prob_true, prob_pred, y_score).plot(ax=ax[0, 3])

        ax[0, 4].hist(
            y_score,
            range=(0, 1),
            bins=20,
            label=pred_col,
            color=color,
            cumulative=True,
            density=True,
            histtype="step",
        )
        # ax = pr_display.ax_
    # ax[0][0].legend()
    ax[0][1].legend()

    ax[0][0].axhline(y=prevalence_pos_label, ls="--", lw=0.5, color="0.5")
    ax[0][1].axhline(y=0.8, ls="--", lw=0.5, color="0.5")
    ax[0][1].axhline(y=0.9, ls="--", lw=0.5, color="0.5")
    ax[0, 4].axhline(y=1 - prevalence_pos_label, ls="--", lw=0.5, color="0.5")
    # ax[0][].set_ylim([0.65, 1.05])
    ax[0][1].set_title("Precision Recall")
    # ax[2][0].set_title("recall")
    fig.suptitle(f"{split}, ({label_col}>{label_threshold})")
    fig.tight_layout()


def plot_confusion_matrix(df, label_col, pred_col="final_pred", title=""):
    fig, ax = plt.subplots(1, 2, figsize=(5 * 2, 5))
    report = classification_report(
        y_true=df[label_col],
        y_pred=df[pred_col],
    )
    print(f"{title}\n{report}")
    ConfusionMatrixDisplay.from_predictions(
        y_true=df[label_col], y_pred=df[pred_col], normalize="pred", ax=ax[0]
    )
    ax[0].set_title("Normalize=pred, Precision")
    ConfusionMatrixDisplay.from_predictions(
        y_true=df[label_col], y_pred=df[pred_col], normalize="true", ax=ax[1]
    )
    ax[1].set_title("Normalize=true, Recall")
    fig.suptitle(title)


def plot_all_prediction_analysis(
    model_evals_dir,
    label_col="label_ids",
    pred_cols=["layer_2_dim_64_pRel", "final_pRel"],
    label_threshold=0,
    title="",
):
    predictions_paths = list(model_evals_dir.glob("./*_predictions.csv"))
    for predictions_path in predictions_paths:
        split = predictions_path.name.split("_predictions")[0]
        predictions_path = model_evals_dir / f"{split}_predictions.csv"
        df = pd.read_csv(predictions_path)
        print(f"{split=}, {df.shape=}")
        make_prediction_analysis_plots(
            df,
            pred_cols,
            label_col=label_col,
            split=split,
            label_threshold=label_threshold,
        )
        plot_confusion_matrix(
            df, label_col, pred_col="final_pred", title=f"{title}\n{split}"
        )
