import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from evaluations_plots import plot_res


def plot_diffs():
    res_df = pd.read_csv("results/csv/all_results_fixed_fw.csv")

    # convert use_aug, use_layer_norm, and feats_weighting to meaningful short names
    res_df["use_aug"] = res_df["use_aug"].apply(lambda x: "aug" if x else "n_aug")
    res_df["use_layer_norm"] = res_df["use_layer_norm"].apply(lambda x: "ln" if x else "n_ln")
    res_df["feats_weighting"] = res_df["feats_weighting"].apply(lambda x: "fw" if x else "n_fw")

    res_df["config"] = res_df.apply(
        lambda row: f"{row['dataset']}-{row['use_aug']}-{row['use_layer_norm']}-{row['feats_weighting']}",
        axis=1)

    configs = res_df["config"].values
    full_scores = res_df["score_full_mean"].values
    partial_scores = res_df["score_partial_mean"].values
    raw_ds_names = res_df["dataset"].values

    plot_res(configs, raw_ds_names, raw_ds_names, full_scores, partial_scores)


def plot_res_old(datasets, full_results, partial_results):
    res_df = pd.read_csv("results/csv/all_results_fixed_fw.csv")

    # convert use_aug, use_layer_norm, and feats_weighting to meaningful short names
    res_df["use_aug"] = res_df["use_aug"].apply(lambda x: "aug" if x else "n_aug")
    res_df["use_layer_norm"] = res_df["use_layer_norm"].apply(lambda x: "ln" if x else "n_ln")
    res_df["feats_weighting"] = res_df["feats_weighting"].apply(lambda x: "fw" if x else "n_fw")

    # plot the results as a bar subplots, with each bar being different configurations and each subplot being a dataset - there are 9 runs_names
    # a configuration is a combination of use_aug, use_layer_norm, and feats_weighting
    # create subplots for each dataset, and a color for each configuration

    # create a new column for the configuration
    res_df["config"] = res_df.apply(lambda row: f"{row['use_aug']}-{row['use_layer_norm']}-{row['feats_weighting']}",
                                    axis=1)

    plt.subplots(4, 2, figsize=(20, 20))
    color_palette = plt.get_cmap("tab10").colors[:8]

    for i, (ds_name, ds) in enumerate(res_df.groupby("dataset")):
        plt.subplot(4, 2, i + 1)
        plt.title(ds_name)
        plt.bar(ds["config"], ds["score_partial_mean"] - ds["score_partial_mean"].min(), color=color_palette,
                yerr=ds["score_partial_std"])
        plt.xticks(rotation=45)
        # plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.xlabel("Configuration")

    plt.tight_layout()
    plt.savefig("full_results.png")
    # plt.show()
    plt.clf()

    ### plot 3 more grouped bar plots, each time show the diffs between the scores of the 2 configurations according to:
    # 1. use_aug
    # 2. use_layer_norm
    # 3. feats_weighting

    # 1. use_aug

    plt.subplots(4, 2, figsize=(20, 20))
    color_palette = plt.get_cmap("tab10").colors[:8]

    for i, (ds_name, ds) in enumerate(res_df.groupby("dataset")):
        plt.subplot(4, 2, i + 1)

        legend = set()

        for j, (use_aug, use_aug_ds) in enumerate(ds.groupby("use_aug")):
            plt.bar(np.arange(use_aug_ds.shape[0]) * 3 + j,
                    use_aug_ds["score_partial_mean"] - ds["score_partial_mean"].min(), color=color_palette[j],
                    yerr=use_aug_ds["score_partial_std"])
            plt.xticks(np.arange(use_aug_ds.shape[0]) * 3 + j,
                       [f"{x}-{y}" for x, y in zip(use_aug_ds["use_layer_norm"], use_aug_ds["feats_weighting"])],
                       rotation=45)

            legend.add(f"{use_aug}")

        plt.legend(list(legend))
        plt.title(ds_name)
        plt.ylabel("Score")

    plt.tight_layout()
    plt.savefig("all_results_use_aug.png")
    # plt.show()
    plt.clf()

    # 2. use_layer_norm

    plt.subplots(4, 2, figsize=(20, 20))
    color_palette = plt.get_cmap("tab10").colors[:8]

    for i, (ds_name, ds) in enumerate(res_df.groupby("dataset")):
        plt.subplot(4, 2, i + 1)

        legend = set()

        for j, (use_layer_norm, use_layer_norm_ds) in enumerate(ds.groupby("use_layer_norm")):
            plt.bar(np.arange(use_layer_norm_ds.shape[0]) * 3 + j,
                    use_layer_norm_ds["score_partial_mean"] - ds["score_partial_mean"].min(), color=color_palette[j],
                    yerr=use_layer_norm_ds["score_partial_std"])
            plt.xticks(np.arange(use_layer_norm_ds.shape[0]) * 3 + j,
                       [f"{x}-{y}" for x, y in zip(use_layer_norm_ds["use_aug"], use_layer_norm_ds["feats_weighting"])],
                       rotation=45)

            legend.add(f"{use_layer_norm}")

        plt.legend(list(legend))
        plt.title(ds_name)
        plt.ylabel("Score")

    plt.tight_layout()
    plt.savefig("all_results_use_layer_norm.png")
    # plt.show()
    plt.clf()

    # 3. feats_weighting

    plt.subplots(4, 2, figsize=(20, 20))
    color_palette = plt.get_cmap("tab10").colors[:8]

    for i, (ds_name, ds) in enumerate(res_df.groupby("dataset")):
        plt.subplot(4, 2, i + 1)

        legend = set()

        for j, (feats_weighting, feats_weighting_ds) in enumerate(ds.groupby("feats_weighting")):
            plt.bar(np.arange(feats_weighting_ds.shape[0]) * 3 + j,
                    feats_weighting_ds["score_partial_mean"] - ds["score_partial_mean"].min(), color=color_palette[j],
                    yerr=feats_weighting_ds["score_partial_std"])
            plt.xticks(np.arange(feats_weighting_ds.shape[0]) * 3 + j,
                       [f"{x}-{y}" for x, y in
                        zip(feats_weighting_ds["use_aug"], feats_weighting_ds["use_layer_norm"])],
                       rotation=45)

            legend.add(f"{feats_weighting}")

        plt.legend(list(legend))
        plt.title(ds_name)
        plt.ylabel("Score")

    plt.tight_layout()
    plt.savefig("all_results_feats_weighting.png")
    # plt.show()
    plt.clf()


if __name__ == '__main__':
    plot_diffs()
