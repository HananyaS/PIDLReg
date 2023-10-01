import numpy as np
import xgboost as xgb
from main_experimens import get_split_data
import os
import pandas as pd
from copy import deepcopy


if __name__ == '__main__':
    datasets = os.listdir("data/Tabular")
    datasets.remove("Sensorless")
    datasets.remove("Credit")

    full_results = {}
    partial_results = {}

    for dataset in datasets:
        print("Starting", dataset)
        # if dataset in ["Sensorless", "Credit"]:
        #     continue
        train_ds, val_ds, test_ds,  = get_split_data(dataset, use_aug=True, as_loader=False, norm=False)

        X_train, y_train = train_ds.X, train_ds.y
        X_val, y_val = val_ds.X, val_ds.y
        X_test, y_test = test_ds.X, test_ds.y

        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(X_train, y_train)
        score = xgb_model.score(X_test, y_test)

        full_results[dataset] = score
        print(f"{dataset}: {score}")

        partial_res = []

        for i in range(X_test.shape[1]):
            X_test_copy = deepcopy(X_test)
            X_test_copy[:, i] = 0
            score = xgb_model.score(X_test_copy, y_test)
            partial_res.append(score)

        partial_results[dataset] = f"{np.mean(partial_res)} +- {np.std(partial_res)}"

    pd.DataFrame.from_dict(full_results, orient="index").to_csv("xgb_sota_full.csv")
    pd.DataFrame.from_dict(partial_results, orient="index").to_csv("xgb_sota_partial.csv")