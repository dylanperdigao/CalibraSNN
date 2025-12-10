import os
import warnings
import numpy as np
import optuna
import time

from numpy.random import seed as numpy_seed
from random import seed as random_seed
from torch import manual_seed as torch_seed
from torch.mps import manual_seed as torch_mps_seed
from torch.cuda import manual_seed as torch_cuda_seed
from torch.cuda import manual_seed_all as torch_cuda_all_seed
from torch.backends import cudnn as torch_cudnn
from torch import use_deterministic_algorithms as torch_use_deterministic_algorithms

from optuna.storages import RetryFailedTrialCallback
from optuna.pruners import ThresholdPruner
from optuna.samplers import TPESampler

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")
from modules.models import FFSNN, FFNN, CSNN, CNN
from modules.other.utils import read_data

BASE_SEED = None

# Other
warnings.filterwarnings("ignore")
np.random.seed(BASE_SEED)
PATH = os.path.dirname(os.path.realpath(__file__))
print("PATH",PATH)

def fix_seeds(seed):
    """
    Fix the seeds of the random number generators
    
    Parameters:
        seed: int
            seed number
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    numpy_seed(seed) 
    random_seed(seed) 
    torch_seed(seed) 
    torch_cuda_seed(seed) 
    torch_cuda_all_seed(seed) 
    torch_mps_seed(seed) 
    torch_cudnn.deterministic = True 
    torch_cudnn.benchmark = False
    torch_use_deterministic_algorithms(True) 


def optimize_parameters(trial, dataset_name, train_dfs, test_dfs, hyperparameters):
    """
    Optimize the hyperparameters of the model
    
    Parameters:
        trial: optuna.Trial
            trial object
        dataset_name: str
            name of the dataset
        train_dfs: dict
            dictionary with the training dataframes
        test_dfs: dict
            dictionary with the test dataframes
        hyperparameters: dict
            dictionary with the hyperparameters

    Returns:
        objectives: list
            list with the objectives
    """
    print(f"Trial {trial.number}")
    layers = len(hyperparameters['topology'])
    batch_size = trial.suggest_categorical('batch', [2**i for i in range(8, 11)]) if hyperparameters['batch'] is None else hyperparameters['batch']
    num_epochs = trial.suggest_int('epoch', 1, 10) if hyperparameters['epoch'] is None else hyperparameters['epoch']
    weight_minority_class = trial.suggest_float('weight', 0.95, 1, step=0.00001) if hyperparameters['weight'] is None else hyperparameters['weight']
    adam_betas = tuple(trial.suggest_float(f'adam_beta{i+1}', 0.97, 0.99, step=0.00001) for i in range(2)) if hyperparameters['adam_beta'] is None else hyperparameters['adam_beta']
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, step=0.00001) if hyperparameters['learning_rate'] is None else hyperparameters['learning_rate']
    if "FFNN" in MODEL or "CNN" in MODEL:
        hyperparameters = {
            "name": hyperparameters['name'],
            "topology": hyperparameters['topology'],
            "batch": batch_size,
            "epoch": num_epochs,
            "weight": weight_minority_class,
            "adam_beta": adam_betas,
            "learning_rate": learning_rate
        }
    else:
        num_steps = trial.suggest_int('step', 1, 25, step=1) if hyperparameters['step'] is None else hyperparameters['step']
        betas = tuple(trial.suggest_float(f'beta{i+1}', 0.1, 1, step=0.00001) for i in range(layers)) if hyperparameters['beta'] is None else hyperparameters['beta']
        slope = trial.suggest_int('slope', 10, 90, step=1) if hyperparameters['slope'] is None else hyperparameters['slope']
        thresholds=tuple(trial.suggest_float(f'threshold{i+1}', 0.1, 1, step=0.00001) for i in range(layers)) if hyperparameters['threshold'] is None else hyperparameters['threshold']
        hyperparameters = {
            "name": hyperparameters['name'],
            "topology": hyperparameters['topology'],
            "batch": batch_size,
            "epoch": num_epochs,
            "step": num_steps,
            "beta": betas,
            "slope": slope,
            "threshold": thresholds,
            "weight": weight_minority_class,
            "adam_beta": adam_betas,
            "learning_rate": learning_rate
        }
    train_df = train_dfs[dataset_name].iloc[:, :32]
    test_df = test_dfs[dataset_name].iloc[:, :32]
    x_train = train_df.drop(columns=["fraud_bool"])
    y_train = train_df["fraud_bool"]
    x_test = test_df.drop(columns=["fraud_bool"])
    y_test = test_df["fraud_bool"]
    num_classes = len(np.unique(y_train))
    num_features = len(x_train.columns)
    print(f"Num classes: {num_classes} Num features: {num_features}")
    if MODEL == "FFSNN" or MODEL == "FCSNN":
        model = FFSNN(
            num_features=num_features,
            num_classes=num_classes,
            hyperparameters=hyperparameters,
            gpu_number=int(trial.number)%3,
            verbose=0
        )
    elif MODEL == "FFNN":
        model = FFNN(
            num_features=num_features,
            num_classes=num_classes,
            hyperparameters=hyperparameters,
            gpu_number=int(trial.number)%3,
            verbose=0
        )
    elif MODEL == "CNN":
        model = CNN(
            num_features=num_features,
            num_classes=num_classes,
            hyperparameters=hyperparameters,
            gpu_number=int(trial.number)%3,
            verbose=0
        )
    else: # CSNN
        model = CSNN(
            num_features=num_features,
            num_classes=num_classes,
            hyperparameters=hyperparameters,
            gpu_number=int(trial.number)%3,
            verbose=0
        )
    # Train the model
    fit_time = time.time()
    model.fit(x_train, y_train)
    trial.set_user_attr("@time train", time.time()-fit_time)
    # Evaluate the model
    inference_time = time.time()
    predictions, targets = model.predict(x_test, y_test)
    trial.set_user_attr("@time inference", time.time()-inference_time)
    # ETA
    eta = (TRIALS_OPTUNA*(time.time()-fit_time)-trial.number*(time.time()-fit_time))/3600
    print(f"ETA: {eta/24:.0f}d {eta%24:.0f}h {eta%1*60:.0f}m")
    # Metrics
    metrics = model.evaluate(targets, predictions)
    metrics_aequitas = model.evaluate_business_constraint(targets, predictions)
    metrics.update(metrics_aequitas)
    fairness_age = model.evaluate_fairness(x_test, targets, predictions, "customer_age", 50)
    metrics.update({k+"_age": v for k, v in fairness_age.items()})
    fairness_income = model.evaluate_fairness(x_test, targets, predictions, "income", 0.5)
    metrics.update({k+"_income": v for k, v in fairness_income.items()})
    fairness_employement = model.evaluate_fairness(x_test, targets, predictions, "employment_status", 3)
    metrics.update({k+"_employment": v for k, v in fairness_employement.items()})
    print(f'[{hyperparameters["name"]}] Trial {trial.number}: Recall–{metrics["recall"]*100:.1f}% FPR–{metrics["fpr"]*100:.1f}%') if (metrics["recall"]>0.4 and metrics["fpr"]<0.05) else None
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
    print(f"[{hyperparameters['name']}] Trial {trial.number}: Recall–{metrics['recall']*100:.5f}% FPR–{metrics['fpr']*100:.5f}%")
    objectives = [metrics[y] for (_,y) in OBJECTIVE]
    return objectives

def main(datasets_list, study_name, trials_optuna, sampler, objective, hyperparameters):
    base_path = f"{PATH}/../data/"
    _, datasets, train_dfs, test_dfs = read_data(base_path, datasets_list, seed=BASE_SEED)
    for dataset_name in datasets.keys(): 
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///TEST.db",
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
            pruner=ThresholdPruner(lower=0.01, upper=0.99)
        )
        study.optimize(lambda trial, dataset_name=dataset_name: optimize_parameters(trial, dataset_name, train_dfs, test_dfs, hyperparameters), n_trials=trials_optuna)
        try:
            print(study.best_params)
            print(study.best_value)
            print(study.best_trial)
        except Exception:
            pass



if __name__ == "__main__":
    topology = ("SM_1H_K2", [(1, 16, 2), (16, 32, 2), (224, 2)])
    """
    topology = ("SM_1H_K2", [(1, 16, 2), (16, 32, 2), (224, 2)])
    topology = ("SM_1H_K3", [(1, 16, 3), (16, 32, 3), (224, 3)])
    topology = ("SM_1H_K4", [(1, 16, 4), (16, 32, 4), (224, 4)])
    topology = ("SM_1H_K5", [(1, 16, 5), (16, 32, 5), (224, 5)])
    topology = ("SM_2H_K2", [(1, 16, 2), (16, 32, 2), (32, 64, 2), (192, 2)])
    topology = ("SM_2H_K3", [(1, 16, 3), (16, 32, 3), (32, 64, 3), (192, 3)])
    topology = ("SM_2H_K4", [(1, 16, 4), (16, 32, 4), (32, 64, 4), (192, 4)])
    topology = ("SM_3H_K2", [(1, 16, 2), (16, 32, 2), (32, 64, 2), (64, 128, 2), (128, 2)])
    #
    topology = ("MM_1H_K2", [(1, 32, 2), (32, 64, 2), (448, 2)])
    topology = ("MM_1H_K3", [(1, 32, 3), (32, 64, 3), (448, 3)])
    topology = ("MM_1H_K4", [(1, 32, 4), (32, 64, 4), (448, 4)])
    topology = ("MM_1H_K5", [(1, 32, 5), (32, 64, 5), (448, 5)])
    topology = ("MM_2H_K2", [(1, 32, 2), (32, 64, 2), (64, 128, 2), (384, 2)])
    topology = ("MM_2H_K3", [(1, 32, 3), (32, 64, 3), (64, 128, 3), (384, 3)])
    topology = ("MM_2H_K4", [(1, 32, 4), (32, 64, 4), (64, 128, 4), (384, 4)])
    topology = ("MM_3H_K2", [(1, 32, 2), (32, 64, 2), (64, 128, 2), (128, 256, 2), (512, 2)])
    #
    topology = ("LM_1H_K2", [(1, 64, 2), (64, 128, 2), (896, 2)])
    topology = ("LM_1H_K3", [(1, 64, 3), (64, 128, 3), (896, 3)])
    topology = ("LM_1H_K4", [(1, 64, 4), (64, 128, 4), (896, 4)])
    topology = ("LM_1H_K5", [(1, 64, 5), (64, 128, 5), (896, 5)])
    topology = ("LM_2H_K2", [(1, 64, 2), (64, 128, 2), (128, 256, 2), (768, 2)])
    topology = ("LM_2H_K3", [(1, 64, 3), (64, 128, 3), (128, 256, 3), (768, 3)])
    topology = ("LM_2H_K4", [(1, 64, 4), (64, 128, 4), (128, 256, 4), (768, 4)])
    topology = ("LM_3H_K2", [(1, 64, 2), (64, 128, 2), (128, 256, 2), (256, 512, 2), (512, 2)])
    #
    topology = ("CNN_1H", [(1, 64, 2), (64, 128, 2), (896, 2)])
    topology = ("CNN_2H", [(1, 64, 2), (64, 128, 2), (128, 256, 2), (768 , 2)])
    topology = ("CNN_3H", [(1, 64, 2), (64, 128, 2), (128, 256, 2), (256, 512, 2), (512, 2)])
    #
    topology = ("FFSNN_1H", [64, 128, 2])
    topology = ("FFSNN_2H", [64, 128, 256, 2])
    topology = ("FFSNN_3H", [64, 128, 256, 512, 2])
    #
    topology = ("FFNN_1H", [64, 128, 2])
    topology = ("FFNN_2H", [64, 128, 256, 2])
    topology = ("FFNN_3H", [64, 128, 256, 512, 2])
    """
    MODEL = topology[0].split("_")[0]
    HYPERPARAMETERS = {
        "name": topology[0], 
        "topology": topology[1],
        "batch": None,
        "epoch": None,
        "step": None,
        "beta": None,
        "slope": None,
        "threshold": None,
        "weight": None,
        "adam_beta": None,
        "learning_rate": None
    }
    DATASETS = ["Base"]
    STUDY_NAME = f"TEST-{HYPERPARAMETERS['name']}"
    TRIALS_OPTUNA = 1050
    SAMPLER = TPESampler()
    OBJECTIVE = [("minimize","fpr"), ("maximize","recall")]
    main(DATASETS, STUDY_NAME, TRIALS_OPTUNA, SAMPLER, OBJECTIVE, HYPERPARAMETERS)

