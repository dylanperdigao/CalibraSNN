import numpy as np
import os
import time 

from datetime import datetime
from pyJoules.device import Device, DeviceFactory
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.energy_meter import EnergyMeter

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from modules.other.random_search import RandomValueTrial
from modules.other.utils import read_data, experimental_print
from modules.other import hyperparameters
from modules.models import  FFSNN, CSNN, FFNN, CNN

DATASET_LIST = ["Base", "Variant I", "Variant II","Variant III", "Variant IV", "Variant V"]
NUM_TRIALS = 100
BEGIN_TRIAL = 0
BASE_SEED = 42

PATH = os.path.dirname(os.path.realpath(__file__))
METRICS_ENERGY = ["time_train","time_test","energy_train", "energy_test"]
METRICS_NAME_GLOBAL = ["accuracy", "precision", "recall", "fpr", "f1_score","auc"]
METRICS_NAME_5FPR = ["accuracy@5FPR","precision@5FPR", "recall@5FPR", "fpr@5FPR", "f1_score@5FPR"]
METRICS_FAIRNESS = ["fpr_ratio_age", "fpr_ratio_income", "fpr_ratio_employment"]

MODEL = "FFSNN"
GPU_NUMBER = 2
HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_CNN_1H
#HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_CNN_2H
##HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_CNN_3H

##HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_FFNN_1H
##HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_FFNN_2H
##HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_FFNN_3H

##HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_FFSNN_1H
##HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_FFSNN_2H
##HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_FFSNN_3H

##HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_SM_1H
##HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_SM_2H
##HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_SM_3H
##HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_MM_1H
##HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_MM_2H
##HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_MM_3H
##HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_LM_1H
##HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_LM_2H
##HYPERPARAMETERS = hyperparameters.HYPERPARAMETERS_LM_3H

EXPERIMENT_NAME = f"ECML2025-{HYPERPARAMETERS['architecture']}-{NUM_TRIALS}trials-begin{BEGIN_TRIAL}"
FIXED_DATE = None 
MEASURE_ENERGY = True


def dataset_loop(train_dfs, test_dfs, dataset_name, trial_number, seed, path, runs):
    x_train = train_dfs[dataset_name].drop(columns=["fraud_bool"])
    y_train = train_dfs[dataset_name]["fraud_bool"]
    x_test = test_dfs[dataset_name].drop(columns=["fraud_bool"])
    y_test = test_dfs[dataset_name]["fraud_bool"]
    num_classes = len(np.unique(y_train))
    num_features = len(x_train.columns)
    if MODEL == "FFSNN" or MODEL == "FCSNN":
        model = FFSNN(
            num_features=num_features,
            num_classes=num_classes,
            hyperparameters=HYPERPARAMETERS,
            gpu_number=GPU_NUMBER,
            verbose=0
        )
    elif MODEL == "CSNN":
        model = CSNN(
            num_features=num_features,
            num_classes=num_classes,
            hyperparameters=HYPERPARAMETERS,
            gpu_number=GPU_NUMBER,
            verbose=0
        )
    elif MODEL == "FFNN":
        model = FFNN(
            num_features=num_features,
            num_classes=num_classes,
            hyperparameters=HYPERPARAMETERS,
            gpu_number=GPU_NUMBER,
            verbose=0
        )
    elif MODEL == "CNN":
        model = CNN(
            num_features=num_features,
            num_classes=num_classes,
            hyperparameters=HYPERPARAMETERS,
            gpu_number=GPU_NUMBER,
            verbose=0
        )
    if MEASURE_ENERGY:
        # Energy
        domains = [NvidiaGPUDomain(0)]
        devices = DeviceFactory.create_devices(domains)
        meter = EnergyMeter(devices)
        # train
        meter.start()
        model.fit(x_train, y_train)
        meter.stop()
        trace_train = meter.get_trace()
        # test
        meter.start()
        predictions, targets = model.predict(x_test, y_test)
        meter.stop()
        trace_test = meter.get_trace()
        # store energy metrics
        energy_metrics = {
            "time_train": trace_train[0].duration,
            "time_test": trace_test[0].duration,
            "energy_train": sum(trace_train[0].energy.values()) / 1000,
            "energy_test": sum(trace_test[0].energy.values()) / 1000
        }
    else:
        # train
        t = time.time()
        model.fit(x_train, y_train)
        t_train = time.time() - t
        # test
        t = time.time()
        predictions, targets = model.predict(x_test, y_test)
        t_test = time.time() - t
        # store energy metrics
        energy_metrics = {
            "time_train": t_train,
            "time_test": t_test,
            "energy_train": 0,
            "energy_test": 0
        }
    metrics = model.evaluate(targets, predictions)
    metrics_constraint = model.evaluate_business_constraint(targets, predictions)
    metrics.update(metrics_constraint)
    fairness_age = model.evaluate_fairness(x_test, targets, predictions, "customer_age", 50)
    metrics.update({k+"_age": v for k, v in fairness_age.items()})
    fairness_income = model.evaluate_fairness(x_test, targets, predictions, "income", 0.5)
    metrics.update({k+"_income": v for k, v in fairness_income.items()})
    fairness_employement = model.evaluate_fairness(x_test, targets, predictions, "employment_status", 3)
    metrics.update({k+"_employment": v for k, v in fairness_employement.items()})
    results = {}
    results["dataset"] = dataset_name
    results["trial"] = trial_number
    results["seed"] = seed
    for metric in METRICS_ENERGY:
        results[metric] = energy_metrics[metric]
    for metric in METRICS_NAME_GLOBAL:
        results[metric] = metrics[metric]
    for metric in METRICS_NAME_5FPR:
        results[metric] = metrics_constraint[metric]
    for metric in METRICS_FAIRNESS:
        results[metric] = metrics[metric]
    csv_row = ','.join([str(x) for x in results.values()])
    with open(path, "a") as f:
        f.write(f"{csv_row}\n")
    prev_runs = runs.get(dataset_name, [])
    prev_runs.append(results)
    runs[dataset_name] = prev_runs
    try:
        experimental_print(f"[{HYPERPARAMETERS['architecture']}] TPR@5FPR: {results['recall@5FPR']:.2f} – PE age: {results['fpr_ratio_age']:.2f}")
    except:
        pass
    return runs

def simulation(datasets, train_dfs, test_dfs, path="./results.csv"):
    np.random.seed(BASE_SEED)
    seeds = np.random.choice(list(range(1_000_000)), size=NUM_TRIALS, replace=False)
    runs = {}
    for trial in range(NUM_TRIALS):
        seed = seeds[trial]
        trial_number = trial
        trial = RandomValueTrial(seed=seed)
        if trial_number < BEGIN_TRIAL:
            print(f"Skipping trial {trial_number} – seed {seed}")
            continue
        for dataset_name in datasets.keys():
            print(f"Running trial {trial_number} with seed {seed} on dataset {dataset_name}")
            dataset_loop(train_dfs, test_dfs, dataset_name, trial_number, seed, path, runs) 

def main():
    base_path = f"{PATH}/../../data/"
    _, datasets, train_dfs, test_dfs = read_data(base_path, DATASET_LIST)
    if not FIXED_DATE:
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        date = FIXED_DATE
    experiment_dir = f"{PATH}/results/"
    results_path = f"{experiment_dir}/{date}-{EXPERIMENT_NAME}.csv"
    os.makedirs(experiment_dir, exist_ok=True)
    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            f.write("dataset,trial,seed," + ",".join(METRICS_ENERGY) + "," + ",".join(METRICS_NAME_GLOBAL) + "," + ",".join(METRICS_NAME_5FPR) + "," + ",".join(METRICS_FAIRNESS) + "\n")
    simulation(datasets, train_dfs, test_dfs, path=results_path)
    
if __name__ == "__main__":
    main()
