import lightgbm as lgbm  
import os
import warnings
import numpy as np
import optuna
import time      

from optuna.storages import RetryFailedTrialCallback
from optuna.pruners import ThresholdPruner
from optuna.samplers import TPESampler

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")
from modules.other.utils import read_data
from modules.metrics import evaluate, evaluate_fairness, evaluate_business_constraint

BASE_SEED = None

# Other
warnings.filterwarnings("ignore")
np.random.seed(BASE_SEED)
PATH = os.path.dirname(os.path.realpath(__file__))
print("PATH",PATH)

def optimize_parameters(trial, dataset_name, train_dfs, test_dfs):
    print(f"Trial {trial.number} – GPU {int(trial.number)%3}")
    n_estimators = trial.suggest_int("n_estimators", 20, 10000, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 30)
    learning_rate = trial.suggest_float("learning_rate", 0.02, 0.1, log=True)
    num_leaves = trial.suggest_int("num_leaves", 10, 100, log=True)
    boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "goss"])
    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 5, 200, log=True)
    max_bin = trial.suggest_int("max_bin", 100, 500)
    enable_bundle = trial.suggest_categorical("enable_bundle", [True, False])
    hyperparameters = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "boosting_type": boosting_type,
        "min_data_in_leaf": min_data_in_leaf,
        "max_bin": max_bin,
        "enable_bundle": enable_bundle
    }

    train_df = train_dfs[dataset_name].iloc[:, :32]
    test_df = test_dfs[dataset_name].iloc[:, :32]

    x_train = train_df.drop(columns=["fraud_bool"])
    y_train = train_df["fraud_bool"]
    x_test = test_df.drop(columns=["fraud_bool"])
    y_test = test_df["fraud_bool"]

    model = lgbm.LGBMClassifier(n_jobs=5, **hyperparameters, verbose=-1) 

    fit_time = time.time()
    model.fit(x_train, y_train, categorical_feature=["payment_type","employment_status","housing_status","source","device_os"])
    trial.set_user_attr("@time train", time.time()-fit_time)
    inference_time = time.time()
    predictions = model.predict(x_test)
    trial.set_user_attr("@time inference", time.time()-inference_time)
    eta = (TRIALS_OPTUNA*(time.time()-fit_time)-trial.number*(time.time()-fit_time))/3600
    print(f"ETA: {eta/24:.0f}d {eta%24:.0f}h {eta%1*60:.0f}m")
    metrics = evaluate(y_test, predictions)
    metrics_aequitas = evaluate_business_constraint(y_test, predictions)
    metrics.update(metrics_aequitas)
    fairness_age = evaluate_fairness(x_test, y_test, predictions, "customer_age", 50)
    metrics.update({k+"_age": v for k, v in fairness_age.items()})
    fairness_income = evaluate_fairness(x_test, y_test, predictions, "income", 0.5)
    metrics.update({k+"_income": v for k, v in fairness_income.items()})
    fairness_employement = evaluate_fairness(x_test, y_test, predictions, "employment_status", 3)
    metrics.update({k+"_employment": v for k, v in fairness_employement.items()})
    print(f'Trial {trial.number}: Recall–{metrics["recall"]*100:.1f}% FPR–{metrics["fpr"]*100:.1f}%') if (metrics["recall"]>0.4 and metrics["fpr"]<0.1) else None
    trial.set_user_attr("@global accuracy", metrics["accuracy"])
    trial.set_user_attr("@global precision", metrics["precision"])
    trial.set_user_attr("@global recall", metrics["recall"])
    trial.set_user_attr("@global fpr", metrics["fpr"])
    trial.set_user_attr("@global f1_score", metrics["f1_score"])
    trial.set_user_attr("@global auc", metrics["auc"])
    try:
        trial.set_user_attr("@5FPR fpr", metrics["fpr@5FPR"])
        trial.set_user_attr("@5FPR recall", metrics["recall@5FPR"])
        trial.set_user_attr("@5FPR accuracy", metrics["accuracy@5FPR"])
        trial.set_user_attr("@5FPR precision", metrics["precision@5FPR"])
        trial.set_user_attr("@5FPR fpr_ratio_age", metrics["fpr_ratio_age"])
        trial.set_user_attr("@5FPR fpr_ratio_income", metrics["fpr_ratio_income"])
        trial.set_user_attr("@5FPR fpr_ratio_employment", metrics["fpr_ratio_employment"])
        trial.set_user_attr("@5FPR threshold", metrics["threshold"])
    except Exception:
        pass
    return [metrics[y] for (_,y) in OBJECTIVE]

def main(datasets_list, study_name, trials_optuna, sampler, objective):
    base_path = f"{PATH}/../data/"
    _, datasets, train_dfs, test_dfs = read_data(base_path, datasets_list, seed=BASE_SEED)
    for dataset_name in datasets.keys(): 
        storage = optuna.storages.RDBStorage(
            url="sqlite:///TEST.db",
            heartbeat_interval=60,
            grace_period=120,
            failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
        ) 
        study = optuna.create_study(
            directions=[x for (x,_) in objective],
            storage=storage,
            load_if_exists=True,
            study_name=f"{study_name}",
            sampler=sampler,
            pruner=ThresholdPruner(lower=0.1, upper=0.99)
        )
        study.optimize(lambda trial, dataset_name=dataset_name: optimize_parameters(trial, dataset_name, train_dfs, test_dfs), n_trials=trials_optuna)
        try:
            print(study.best_params)
            print(study.best_value)
            print(study.best_trial)
        except Exception:
            pass



if __name__ == "__main__":
    
    DATASETS = ["Base"]
    STUDY_NAME = f"TEST-LightGBM-{DATASETS[0]}"
    TRIALS_OPTUNA = 1000
    SAMPLER = TPESampler()
    OBJECTIVE = [("minimize","fpr"), ("maximize","recall")]
    
main(DATASETS, STUDY_NAME, TRIALS_OPTUNA, SAMPLER, OBJECTIVE)
