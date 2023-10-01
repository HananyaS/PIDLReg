import os
import json
import argparse
import warnings
from itertools import product, combinations

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.load_data import load_data, split_X_y
from regModel import RegModel as Model
from dataset import Dataset
from evaluations_plots import load_xgb_res

from ts_model import TSFrameworkMS, TSFrameworkOS

from xgboost import XGBClassifier

from matplotlib import pyplot as plt

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier


warnings.filterwarnings("ignore")

PROJECT_DIR = "."
os.chdir(PROJECT_DIR)

parser = argparse.ArgumentParser()

str2bool = lambda x: (str(x).lower() == "true")

parser.add_argument("--dataset", type=str, default="toy")
parser.add_argument("--full", type=str2bool, default=True)
parser.add_argument("--verbose", type=str2bool, default=False)

parser.add_argument("--reg_type", type=str, default="var")
parser.add_argument("--weight_type", type=str, default="loss")
parser.add_argument("--alpha", type=float, default=0)
parser.add_argument("--use_layer_norm", type=str2bool, default=False)
parser.add_argument("--use_aug", type=str2bool, default=False)

parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dropout", type=float, default=0)

parser.add_argument("--n_epochs", type=int, default=300)
parser.add_argument("--early_stopping", type=int, default=30)

parser.add_argument("--resfile", type=str, default="tmp.csv")

parser.add_argument("--run_grid", type=str2bool, default=False)
parser.add_argument("--run_ts", type=str2bool, default=False)
parser.add_argument("--full_test", type=str2bool, default=True)

parser.add_argument("--load_params", type=str2bool, default=True)
parser.add_argument("--show_figs", type=str2bool, default=False)

parser.add_argument("--toy_model", type=str2bool, default=False)
parser.add_argument("--plot_dir", type=str, default="removal_feats_plots")

parser.add_argument("--k", type=int, default=3)
parser.add_argument("--N", type=int, default=1000)

args = parser.parse_args()

if "toy" in args.dataset.lower():
    args.toy_model = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
XGB_FULL_RES, XGB_PARTIAL_RES = load_xgb_res()

assert args.reg_type in ["l1", "l2", "max", "var"]
assert args.weight_type in ["loss", "avg", "mult"]
assert args.resfile is None or args.resfile.endswith(
    ".csv"
), "resfile must be None or end with .csv"


def run_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    dataset_name: str,
    **kwargs,
):
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Dataset:\t", dataset_name)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    model = Model(
        train_loader.dataset.X.shape[1],
        int(train_loader.dataset.X.shape[1] * 0.75),
        len(train_loader.dataset.y.unique()),
        use_layer_norm=kwargs.pop("use_layer_norm", True),
        dropout=kwargs.pop("dropout", 0.5),
    )

    model.fit(train_loader, val_loader, dataset_name=dataset_name, **kwargs)

    val_X = val_loader.dataset.X
    val_y = val_loader.dataset.y

    val_model_score_full = model.score(val_X, val_y)

    print()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Model alphas:\n", [a.item() for a in model.alphas])
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    val_scores_partial = []

    for j in range(val_X.shape[1]):
        new_val_X = val_X.clone()
        new_val_X[:, j] = model.alphas[j]

        val_score = model.score(new_val_X, val_y)
        val_scores_partial.append(val_score)

    train_X = train_loader.dataset.X
    train_y = train_loader.dataset.y

    train_model_score_full = model.score(train_X, train_y)

    train_scores_partial = []

    for j in range(train_X.shape[1]):
        new_train_X = train_X.clone()
        new_train_X[:, j] = model.alphas[j]

        train_score = model.score(new_train_X, train_y)
        train_scores_partial.append(train_score)

    return (
        model,
        (
            train_model_score_full,
            np.mean(train_scores_partial).round(3),
            np.std(train_scores_partial).round(3),
        ),
        (
            val_model_score_full,
            np.mean(val_scores_partial).round(3),
            np.std(val_scores_partial).round(3),
        ),
    )


def get_split_data(
    dataset: str,
    use_aug: bool = False,
    norm: bool = True,
    as_loader: bool = True,
    as_df: bool = False,
    batch_size: int = 32,
):
    data = load_data(dataset, args.full)

    if data[-1] == "tabular":
        train, val, test, target_col = data[:-1]
        edges = None

    else:
        train, val, test, edges, target_col = data[:-1]

    train_X, train_y = split_X_y(train, target_col)
    val_X, val_y = split_X_y(val, target_col)
    test_X, test_y = split_X_y(test, target_col)

    train_ds = Dataset(train_X, train_y, norm=False, add_aug=False)
    val_ds = Dataset(val_X, val_y, norm=False, add_aug=False)
    test_ds = Dataset(test_X, test_y, norm=False, add_aug=False)

    if norm:
        mean, std = train_ds.norm()
        val_ds.norm(mean, std)
        test_ds.norm(mean, std)

    if use_aug:
        train_ds.add_augmentations()

    if as_df:
        train_X = train_ds.X
        train_y = train_ds.y
        val_X = val_ds.X
        val_y = val_ds.y
        test_X = test_ds.X
        test_y = test_ds.y

        return train_X, train_y, val_X, val_y, test_X, test_y

    if not as_loader:
        return train_ds, val_ds, test_ds

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def main(resfile: str = "full_results.csv"):
    datasets = os.listdir("data/Tabular")
    datasets.remove("Sensorless")
    datasets.remove("Credit")
    datasets.append("Credit")
    # runs_names.append("Sensorless")

    # runs_names = ["Ecoli", "Accent", "Iris"]

    res = []

    if not os.path.isfile(resfile):
        if os.path.dirname(resfile) != "":
            os.makedirs(os.path.dirname(resfile), exist_ok=True)
        f = open(resfile, "w")
        f.write(
            ",".join(
                [
                    "dataset",
                    "use_aug",
                    "use_layer_norm",
                    "feats_weighting",
                    "score_full_mean",
                    "score_partial_mean",
                    "score_partial_std",
                ]
            )
            + "\n"
        )

    else:
        f = open(resfile, "a")

    for dataset, use_aug, use_layer_norm, feats_weighting in product(
        datasets, [True, False], [True, False], [True, False]
    ):
        if dataset.lower() in ["wine", "htru2", "parkinson"]:
            continue

        train_loader, val_loader, test_loader = get_split_data(dataset, use_aug)
        model, (score_full, score_partial_mean, score_partial_std) = run_model(
            train_loader,
            val_loader,
            dataset_name=dataset,
            use_layer_norm=use_layer_norm,
            feats_weighting=feats_weighting,
        )

        res.append(
            [
                dataset,
                use_aug,
                use_layer_norm,
                feats_weighting,
                score_full,
                score_partial_mean,
                score_partial_std,
            ]
        )

        f.write(",".join([str(v) for v in res[-1]]) + "\n")


def main_one_run(
    dataset: str, verbose: bool = False, full_test: bool = False, **kwargs
):
    use_aug = kwargs.pop("use_aug", False)
    batch_size = kwargs.pop("batch_size", 32)

    if not args.toy_model:
        train_loader, val_loader, test_loader = get_split_data(
            dataset, use_aug=use_aug, batch_size=batch_size
        )

    if args.toy_model:
        N, k = args.N, args.k

        cov = np.eye(k) * 50

        X = np.random.multivariate_normal(np.zeros(k), cov, size=N)

        if "linear" in dataset.lower():
            dataset = f"toy_linear_{N}_{k}"
            print("~~~~~~~~~~~~~~~~ Linear Toy Model ~~~~~~~~~~~~~~~~")
            y = (X.sum(axis=1) > 0).astype(int)

        else:
            dataset = f"toy_non_linear_{N}_{k}"

            print("~~~~~~~~~~~~~~~~ Non-Linear Toy Model ~~~~~~~~~~~~~~~~")
            y = ((X ** 3).sum(axis=1) > 0).astype(int)

        train_X = pd.DataFrame(X[: int(0.8 * X.shape[0])])
        train_y = pd.Series(y[: int(0.8 * X.shape[0])])

        val_X = pd.DataFrame(X[int(0.8 * X.shape[0]) : int(0.9 * X.shape[0])])
        val_y = pd.Series(y[int(0.8 * X.shape[0]) : int(0.9 * X.shape[0])])

        test_X = pd.DataFrame(X[int(0.9 * X.shape[0]) :])
        test_y = pd.Series(y[int(0.9 * X.shape[0]) :])

        train_ds = Dataset(train_X, train_y, norm=False, add_aug=False)
        val_ds = Dataset(val_X, val_y, norm=False, add_aug=False)
        test_ds = Dataset(test_X, test_y, norm=False, add_aug=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    (
        model,
        (train_full_score, train_partial_score_mean, train_partial_score_std),
        (val_full_score, val_partial_score_mean, val_partial_score_std),
    ) = run_model(
        train_loader,
        val_loader,
        dataset_name=dataset,
        **kwargs,
        verbose=verbose,
    )

    if not full_test:
        print(f"Train full score:\t{train_full_score}")
        print(
            f"Train partial score:\t{train_partial_score_mean} +- {train_partial_score_std}"
        )
        print(f"Val full score:\t{val_full_score}")
        print(
            f"Val partial score:\t{val_partial_score_mean} +- {val_partial_score_std}"
        )

        if dataset in XGB_FULL_RES.keys() and dataset in XGB_PARTIAL_RES.keys():
            print(f"XGBoost full score:\t{XGB_FULL_RES[dataset]}")
            print(f"XGBoost partial score:\t{XGB_PARTIAL_RES[dataset]}")

    if full_test:
        xgb = XGBClassifier()
        xgb.fit(train_loader.dataset.X, train_loader.dataset.y)

        rf = RandomForestClassifier()
        rf.fit(train_loader.dataset.X, train_loader.dataset.y)

        if not args.toy_model:
            train_loader_no_aug, val_loader_no_aug, test_loader_no_aug = get_split_data(
                dataset, use_aug=False, batch_size=batch_size
            )

        else:
            train_loader_no_aug = train_loader
            val_loader_no_aug = val_loader

        dn_kwargs = {
            "reg_type": "l2",
            "weight_type": "None",
            "alpha": 0,
            "use_layer_norm": False,
            "lr": kwargs["lr"],
            "dropout": 0,
        }

        dn_model, *_ = run_model(
            train_loader_no_aug,
            val_loader_no_aug,
            dataset_name=dataset,
            **dn_kwargs,
            verbose=False,
        )

        ada = AdaBoostClassifier()
        ada.fit(train_loader.dataset.X, train_loader.dataset.y)

        lgbm = LGBMClassifier()
        lgbm.fit(train_loader.dataset.X, train_loader.dataset.y)

        run_full_test(
            model,
            dn_model,
            xgb,
            rf,
            ada,
            lgbm,
            train_loader,
            test_loader,
            dataset_name=dataset,
        )

    return (
        (train_full_score, train_partial_score_mean, train_partial_score_std),
        (
            val_full_score,
            val_partial_score_mean,
            val_partial_score_std,
        ),
    )


def run_many_datasets(datasets: list = None, **kwargs):
    if datasets is None:
        datasets = os.listdir("data/Tabular")
        datasets.remove("Sensorless")
        datasets.remove("Credit")
        datasets.remove("HTRU2")
        datasets.remove("Ecoli")

    all_res = {}

    for dataset in datasets:
        res = main_one_run(dataset, **kwargs)
        all_res[dataset] = res

    return all_res


def run_grid_search(dataset: str, search_space: dict, resfile: str, **kwargs):
    all_confs = list(product(*search_space.values()))

    if os.path.isfile(resfile):
        prev_file = pd.read_csv(resfile)
        last_conf_num = prev_file["conf_num"].max()

    else:
        last_conf_num = -1

    print("last_conf_num:", last_conf_num)

    len_all_confs = len(all_confs)

    for i, vals in enumerate(all_confs):
        if i < last_conf_num:
            print(f"Skipping conf {i + 1}/{len_all_confs}!")
            continue

        kwargs.update({k: v for k, v in zip(search_space.keys(), vals)})

        (train_full_score, train_partial_score_mean, train_partial_score_std), (
            val_full_score,
            val_partial_score_mean,
            val_partial_score_std,
        ) = main_one_run(dataset, resfile=resfile, **kwargs, full_test=False)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Results saved to:", resfile)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        if not os.path.isfile(resfile):
            f = open(resfile, "a")
            f.write(
                ",".join(
                    [
                        "dataset",
                        "conf_num",
                        *list(search_space.keys()),
                        "train_full_score",
                        "train_partial_score",
                        "val_full_score",
                        "val_partial_score",
                    ]
                )
                + "\n"
            )
            f.close()

        with open(resfile, "a") as f:
            f.write(
                ",".join(
                    [
                        dataset,
                        str(i + 1),
                        *[str(v) for v in vals],
                        str(train_full_score),
                        f"{train_partial_score_mean} +- {train_partial_score_std}",
                        str(val_full_score),
                        f"{val_partial_score_mean} +- {val_partial_score_std}",
                    ]
                )
                + "\n"
            )

        print(f"Done conf {i + 1}/{len_all_confs}!")


def run_full_test(
    model: nn.Module,
    dn: nn.Module,
    xgb: XGBClassifier,
    random_forest: RandomForestClassifier,
    ada: AdaBoostClassifier,
    lgbm: LGBMClassifier,
    train_loader: DataLoader,
    test_loader: DataLoader,
    dataset_name: str,
    max_combinations: int = 60,
    plot: bool = True,
):
    model.eval()
    dn.eval()

    train_X, train_y = train_loader.dataset.X, train_loader.dataset.y
    test_X, test_y = test_loader.dataset.X, test_loader.dataset.y

    n_feats_to_train_with = range(1, test_X.shape[1] + 1)

    model_scores = []
    xgboost_scores = []
    dn_scores = []
    random_forest_scores = []
    ada_scores = []
    lgbm_scores = []
    knn_all_feat_scores = []
    knn_pre_feat_scores = []

    for n_feats in n_feats_to_train_with:
        model_scores_per_n_feat = []
        xgboost_scores_per_n_feat = []
        dn_scores_per_n_feat = []
        random_forest_scores_per_n_feat = []
        ada_scores_per_n_feat = []
        lgbm_scores_per_n_feat = []
        knn_all_feat_scores_per_n_feat = []
        knn_pre_feat_scores_per_n_feat = []

        print(f"Removing {test_X.shape[1] - n_feats}/{test_X.shape[1]} features")


        if test_X.shape[1] < 25:
            all_combs = list(
                combinations(range(test_X.shape[1]), test_X.shape[1] - n_feats)
            )
            np.random.shuffle(all_combs)
            all_combs = all_combs[:max_combinations]

        else:
            all_combs = range(test_X.shape[1])

        for comb in all_combs:
            train_X_clone = train_X.clone()
            test_X_clone = test_X.clone()

            if test_X.shape[1] >= 25:
                random_feats_to_remove = np.random.choice(
                    test_X.shape[1], test_X.shape[1] - n_feats, replace=False
                ).flatten()

            else:
                random_feats_to_remove = np.array(comb)

            test_X_clone[:, random_feats_to_remove] = np.nan

            xgboost_scores_per_n_feat.append(xgb.score(test_X_clone, test_y))

            test_X_clone[:, random_feats_to_remove] = 0

            model_scores_per_n_feat.append(model.score(test_X_clone, test_y))
            dn_scores_per_n_feat.append(dn.score(test_X_clone, test_y))
            random_forest_scores_per_n_feat.append(random_forest.score(test_X_clone, test_y))
            ada_scores_per_n_feat.append(ada.score(test_X_clone, test_y))
            lgbm_scores_per_n_feat.append(lgbm.score(test_X_clone, test_y))

            scaler = StandardScaler()
            scaler.fit(train_X_clone)

            train_X_clone = scaler.transform(train_X_clone)
            test_X_clone = scaler.transform(test_X_clone)

            knn_all_feat = KNeighborsClassifier(n_neighbors=10).fit(
                train_X_clone, train_y
            )
            knn_all_feat_scores_per_n_feat.append(
                knn_all_feat.score(test_X_clone, test_y)
            )

            if len(random_feats_to_remove) > 0:
                train_X_clone[:, random_feats_to_remove] = 0

            knn_pre_feat = KNeighborsClassifier(n_neighbors=10).fit(
                train_X_clone, train_y
            )
            knn_pre_feat_scores_per_n_feat.append(
                knn_pre_feat.score(test_X_clone, test_y)
            )

            if len(random_feats_to_remove) == 0:
                assert np.allclose(
                    knn_all_feat_scores_per_n_feat[-1],
                    knn_pre_feat_scores_per_n_feat[-1],
                )

        model_scores.append(
            (np.mean(model_scores_per_n_feat), np.std(model_scores_per_n_feat))
        )

        dn_scores.append((np.mean(dn_scores_per_n_feat), np.std(dn_scores_per_n_feat)))

        xgboost_scores.append(
            (np.mean(xgboost_scores_per_n_feat), np.std(xgboost_scores_per_n_feat))
        )

        random_forest_scores.append(
            (
                np.mean(random_forest_scores_per_n_feat),
                np.std(random_forest_scores_per_n_feat),
            )
        )

        ada_scores.append(
            (
                np.mean(ada_scores_per_n_feat),
                np.std(ada_scores_per_n_feat),
            )
        )

        lgbm_scores.append(
            (
                np.mean(lgbm_scores_per_n_feat),
                np.std(lgbm_scores_per_n_feat),
            )
        )

        knn_pre_feat_scores.append(
            (
                np.mean(knn_pre_feat_scores_per_n_feat),
                np.std(knn_pre_feat_scores_per_n_feat),
            )
        )

        knn_all_feat_scores.append(
            (
                np.mean(knn_all_feat_scores_per_n_feat),
                np.std(knn_all_feat_scores_per_n_feat),
            )
        )

    if plot:
        n_feats_to_train_with = list(n_feats_to_train_with)[::-1]

        c = 0

        model_scores_mean = [s[0] for s in model_scores[::-1]] + np.random.rand(
            len(n_feats_to_train_with)
        ) * c
        model_scores_std = [s[1] for s in model_scores[::-1]] + np.random.rand(
            len(n_feats_to_train_with)
        ) * c

        dn_scores_mean = [s[0] for s in dn_scores][::-1] + np.random.rand(
            len(n_feats_to_train_with)
        ) * c
        dn_scores_std = [s[1] for s in dn_scores][::-1] + np.random.rand(
            len(n_feats_to_train_with)
        ) * c

        xgboost_scores_mean = [s[0] for s in xgboost_scores][::-1] + np.random.rand(
            len(n_feats_to_train_with)
        ) * c
        xgboost_scores_std = [s[1] for s in xgboost_scores][::-1] + np.random.rand(
            len(n_feats_to_train_with)
        ) * c

        random_forest_scores_mean = [s[0] for s in random_forest_scores][
            ::-1
        ] + np.random.rand(len(n_feats_to_train_with)) * c
        random_forest_scores_std = [s[1] for s in random_forest_scores][
            ::-1
        ] + np.random.rand(len(n_feats_to_train_with)) * c

        ada_scores_mean = [s[0] for s in ada_scores][::-1] + np.random.rand(
            len(n_feats_to_train_with)
        ) * c

        ada_scores_std = [s[1] for s in ada_scores][::-1] + np.random.rand(
            len(n_feats_to_train_with)
        ) * c

        lgbm_scores_mean = [s[0] for s in lgbm_scores][::-1] + np.random.rand(
            len(n_feats_to_train_with)
        ) * c

        lgbm_scores_std = [s[1] for s in lgbm_scores][::-1] + np.random.rand(
            len(n_feats_to_train_with)
        ) * c

        knn_pre_feat_scores_mean = [s[0] for s in knn_pre_feat_scores][
            ::-1
        ] + np.random.rand(len(n_feats_to_train_with)) * c

        knn_pre_feat_scores_std = [s[1] for s in knn_pre_feat_scores][
            ::-1
        ] + np.random.rand(len(n_feats_to_train_with)) * c

        all_models = [
            "XGBoost",
            "Random Forsest",
            "AdaBoost",
            "LGBM",
            "Vanilla NN",
            "NN with regularization",
            "KNN",
        ]

        all_results = pd.DataFrame(columns=["n_feats_to_train_with"] + all_models)
        all_results["n_feats_to_train_with"] = n_feats_to_train_with
        all_results["XGBoost"] = xgboost_scores_mean
        all_results["Random Forest"] = random_forest_scores_mean
        all_results["AdaBoost"] = ada_scores_mean
        all_results["LGBM"] = lgbm_scores_mean
        all_results["Vanilla NN"] = dn_scores_mean
        all_results["NN with regularization"] = model_scores_mean
        all_results["KNN"] = knn_pre_feat_scores_mean
        all_results["XGBoost std"] = xgboost_scores_std
        all_results["Random Forest std"] = random_forest_scores_std
        all_results["AdaBoost std"] = ada_scores_std
        all_results["LGBM std"] = lgbm_scores_std
        all_results["Vanilla NN std"] = dn_scores_std
        all_results["NN with regularization std"] = model_scores_std
        all_results["KNN std"] = knn_pre_feat_scores_std


        os.makedirs("full_test_results_csv", exist_ok=True)

        all_results.to_csv(
            f"full_test_results_csv/{dataset_name.lower()}_results.csv", index=False
        )

        # plot only ensemble methods

        plt.clf()

        plt.plot(
            n_feats_to_train_with, random_forest_scores_mean, label="Random Forest"
        )

        plt.fill_between(
            n_feats_to_train_with,
            np.array(random_forest_scores_mean) - np.array(random_forest_scores_std),
            np.array(random_forest_scores_mean) + np.array(random_forest_scores_std),
            alpha=0.2,
        )

        plt.plot(
            n_feats_to_train_with,
            ada_scores_mean,
            label="AdaBoost",
        )

        plt.fill_between(
            n_feats_to_train_with,
            np.array(ada_scores_mean) - np.array(ada_scores_std),
            np.array(ada_scores_mean) + np.array(ada_scores_std),
            alpha=0.2,
        )

        plt.plot(
            n_feats_to_train_with,
            lgbm_scores_mean,
            label="LightGBM",
        )

        plt.fill_between(
            n_feats_to_train_with,
            np.array(lgbm_scores_mean) - np.array(lgbm_scores_std),
            np.array(lgbm_scores_mean) + np.array(lgbm_scores_std),
            alpha=0.2,
        )

        plt.plot(
            n_feats_to_train_with,
            xgboost_scores_mean,
            label="XGBoost",
        )

        plt.fill_between(
            n_feats_to_train_with,
            np.array(xgboost_scores_mean) - np.array(xgboost_scores_std),
            np.array(xgboost_scores_mean) + np.array(xgboost_scores_std),
            alpha=0.2,
        )

        plt.xlabel("Number of features")
        plt.ylabel("Accuracy")

        plt.title(dataset_name.capitalize())

        plt.legend()

        os.makedirs(os.path.join(args.plot_dir, "ensemble", "svg"), exist_ok=True)

        plt.gca().invert_xaxis()
        # plt.savefig(os.path.join(args.plot_dir, "ensemble))
        plt.savefig(
            os.path.join(
                args.plot_dir, "ensemble", "svg", f"{dataset_name.lower()}.svg"
            )
        )

        if args.show_figs:
            plt.show()

        # plot only neural networks

        plt.clf()

        plt.plot(n_feats_to_train_with, dn_scores_mean, label="Neural Network")

        plt.fill_between(
            n_feats_to_train_with,
            np.array(dn_scores_mean) - np.array(dn_scores_std),
            np.array(dn_scores_mean) + np.array(dn_scores_std),
            alpha=0.2,
        )

        plt.plot(
            n_feats_to_train_with,
            model_scores_mean,
            label=f"Neural Network + Regularization",
        )

        plt.fill_between(
            n_feats_to_train_with,
            np.array(model_scores_mean) - np.array(model_scores_std),
            np.array(model_scores_mean) + np.array(model_scores_std),
            alpha=0.2,
        )

        plt.xlabel("Number of features")
        plt.ylabel("Accuracy")

        plt.title(dataset_name.capitalize())

        plt.legend()

        os.makedirs(os.path.join(args.plot_dir, "nn", "svg"), exist_ok=True)

        plt.gca().invert_xaxis()

        plt.savefig(
            os.path.join(args.plot_dir, "nn", "svg", f"{dataset_name.lower()}.svg")
        )

        if args.show_figs:
            plt.show()

        # plot only random forest, xgboost, and neural network

        plt.clf()

        plt.plot(
            n_feats_to_train_with, random_forest_scores_mean, label="Random Forest"
        )

        plt.fill_between(
            n_feats_to_train_with,
            np.array(random_forest_scores_mean) - np.array(random_forest_scores_std),
            np.array(random_forest_scores_mean) + np.array(random_forest_scores_std),
            alpha=0.2,
        )

        plt.plot(
            n_feats_to_train_with,
            xgboost_scores_mean,
            label="XGBoost",
        )

        plt.fill_between(
            n_feats_to_train_with,
            np.array(xgboost_scores_mean) - np.array(xgboost_scores_std),
            np.array(xgboost_scores_mean) + np.array(xgboost_scores_std),
            alpha=0.2,
        )

        plt.plot(
            n_feats_to_train_with,
            model_scores_mean,
            label=f"Neural Network + Regularization",
        )

        plt.fill_between(
            n_feats_to_train_with,
            np.array(model_scores_mean) - np.array(model_scores_std),
            np.array(model_scores_mean) + np.array(model_scores_std),
            alpha=0.2,
        )

        plt.plot(
            n_feats_to_train_with,
            knn_pre_feat_scores_mean,
            label=f"KNN",
        )

        plt.fill_between(
            n_feats_to_train_with,
            np.array(knn_pre_feat_scores_mean) - np.array(knn_pre_feat_scores_std),
            np.array(knn_pre_feat_scores_mean) + np.array(knn_pre_feat_scores_std),
            alpha=0.2,
        )

        plt.xlabel("Number of features")
        plt.ylabel("Accuracy")

        plt.title(dataset_name.capitalize())

        plt.legend()

        os.makedirs(os.path.join(args.plot_dir, "rf_xgb_nn", "svg"), exist_ok=True)

        plt.gca().invert_xaxis()

        plt.savefig(
            os.path.join(
                args.plot_dir, "rf_xgb_nn", "svg", f"{dataset_name.lower()}.svg"
            )
        )

        if args.show_figs:
            plt.show()


def run_teacher_student(dataset: str, ts_type: str = "OS", **kwargs):
    train_loader, val_loader, test_loader = get_split_data(
        dataset, use_aug=False, batch_size=32
    )

    if ts_type == "MS":
        model = TSFrameworkMS(
            train_loader.dataset.X.shape[1],
            int(train_loader.dataset.X.shape[1] * 0.75),
            len(train_loader.dataset.y.unique()),
        )
    else:
        model = TSFrameworkOS(
            train_loader.dataset.X.shape[1],
            int(train_loader.dataset.X.shape[1] * 0.75),
            len(train_loader.dataset.y.unique()),
        )

    test_score_full, test_score_partial_mean, test_score_partial_std = model.fit(
        train_loader, val_loader, test_loader, w_teacher=1
    )

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Full score:\t{test_score_full}")
    print(f"Partial score:\t{test_score_partial_mean} +- {test_score_partial_std}")


def load_params_from_file(filename: str):
    with open(filename, "r") as f:
        params = json.load(f)

    return params


if __name__ == "__main__":
    if args.run_ts:
        run_teacher_student(args.dataset)

    elif args.run_grid:
        assert args.reg_type in ["l1", "l2", "max", "var"]
        assert args.weight_type in ["loss", "avg", "mult"]

        search_space = {
            "reg_type": ["l1", "l2", "max", "var"],
            "weight_type": [None, "loss", "avg"],
            "alpha": [0.5, 1],
            "use_layer_norm": [False],
            "use_aug": [False, True],
            "lr": [0.01, 0.1],
            "batch_size": [32, 64],
        }

        run_grid_search(
            args.dataset, search_space, args.resfile, n_epochs=args.n_epochs
        )

    else:
        if args.load_params:
            params_file = os.path.join(
                "param_files", "json", f"{args.dataset.lower()}_params.json"
            )

            if not os.path.isfile(params_file):
                print(
                    f"Params file for {args.dataset} is not found! Using the default params instead."
                )

                params_file = os.path.join("param_files", "default_params.json")

            print("#########################################################")
            print(f"Using params from {params_file}")
            print("#########################################################")

            params = load_params_from_file(params_file)

            params["feats_weighting"] = params["weight_type"] not in [None, "None"]
            params["dropout"] = (
                params["dropout"] if "dropout" in params else args.dropout
            )

            main_one_run(
                args.dataset,
                **params,
                resfile=args.resfile,
                full_test=args.full_test,
            )

        else:
            main_one_run(
                args.dataset,
                weight_type=args.weight_type,
                verbose=args.verbose,
                reg_type=args.reg_type,
                alpha=args.alpha,
                use_layer_norm=args.use_layer_norm,
                use_aug=args.use_aug,
                feats_weighting=args.weight_type not in [None, "None"],
                lr=args.lr,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                dropout=args.dropout,
                resfile=args.resfile,
                full_test=args.full_test,
            )