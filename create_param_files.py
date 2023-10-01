import os
import json
import pandas as pd

dataset_grid_res = {x.split('_')[0]: x for x in os.listdir('grid_search_results') if x.endswith('.csv') and 'process' not in x}

os.makedirs("param_files", exist_ok=True)

for dataset, grid_res in dataset_grid_res.items():
    df = pd.read_csv(os.path.join('grid_search_results', grid_res))
    df["partial_scores_mean"] = df["partial_score"].apply(lambda x: float(x.split(" +- ")[0]))
    df["diff_full_partial"] = df["full_score"] - df["partial_scores_mean"]
    df.sort_values(by="diff_full_partial", ascending=True, inplace=True, kind='stable')
    df.sort_values(by="partial_scores_mean", ascending=False, inplace=True, kind='stable')
    df.to_csv(os.path.join("param_files", f"{dataset}_params.csv"), index=False)

    # create params dict of reg_type	weight_type	alpha	use_layer_norm	use_aug	lr	batch_size

    params = {
        "reg_type": str(df.iloc[0]["reg_type"]),
        "weight_type": str(df.iloc[0]["weight_type"]),
        "alpha": float(df.iloc[0]["alpha"]),
        "use_layer_norm": bool(df.iloc[0]["use_layer_norm"]),
        "use_aug": bool(df.iloc[0]["use_aug"]),
        "lr": float(df.iloc[0]["lr"]),
        "batch_size": int(df.iloc[0]["batch_size"]),
    }

    with open(os.path.join("param_files", f"{dataset}_params.json"), 'w') as f:
        json.dump(params, f)

