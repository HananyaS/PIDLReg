import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def load_xgb_res():
    full_res_df = pd.read_csv("xgb_sota_full.csv", index_col=0)
    partial_res_df = pd.read_csv("xgb_sota_partial.csv", index_col=0)
    full_res_as_dict = full_res_df.to_dict()["0"]
    partial_res_as_dict = partial_res_df.to_dict()["0"]
    return full_res_as_dict, partial_res_as_dict


def plot_res(runs_names, raw_datasets_names, labels, full_results, partial_results,
             savefile: str = 'all_results_diffs.png'):
    # raw_datasets_names = [ds.split('-')[0] for ds in runs_names]
    colors = plt.get_cmap("tab20").colors

    xgb_full_res, _ = load_xgb_res()
    diff_full_xgb = {ds: score - xgb_full_res[ds_raw_name] for ds, ds_raw_name, score in
                     zip(runs_names, raw_datasets_names, full_results)}
    diff_partial_full = {ds: full_results[i] - partial_results[i] for i, ds in enumerate(runs_names)}

    # ds2color = {ds: colors[i] for i, ds in enumerate(XGB_RES.keys())}

    # colors = [ds2color[ds] for ds in raw_datasets_names]

    plt.clf()

    for label in np.unique(labels):
        idx = list(filter(lambda i: labels[i] == label, range(len(labels))))
        x = [diff_full_xgb[ds] for ds in np.array(runs_names)[idx]]
        y = [diff_partial_full[ds] for ds in np.array(runs_names)[idx]]

        plt.scatter(x, y, label=label)

    # plt.scatter(list(diff_full_xgb.values()), list(diff_partial_full.values()))

    plt.xlabel("Diff full - XGB")
    plt.ylabel("Diff full - partial")
    plt.title("All results")

    # add annotation for each point

    # for ds in diff_full_xgb.keys():
    #     plt.annotate(ds, (diff_full_xgb[ds], diff_partial_full[ds]))

    plt.legend()
    plt.savefig(savefile)
    plt.show()
