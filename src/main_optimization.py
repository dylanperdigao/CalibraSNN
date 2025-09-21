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

# add modules to the path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../")
from modules.models import ModelSNN, FFSNN, FFNN, CSNN, CNN
from modules.other.utils import read_data, experimental_print

MACHINE_NAME = os.uname()[1] if "crai" in os.uname()[1] else "local"
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
    numpy_seed(seed) # numpy.random.seed(seed)
    random_seed(seed) # random.seed(seed)
    torch_seed(seed) # torch.manual_seed(seed)
    torch_cuda_seed(seed) # torch.cuda.manual_seed(seed)
    torch_cuda_all_seed(seed) # torch.cuda.manual_seed_all(seed)
    torch_mps_seed(seed) # torch.mps.manual_seed(seed)
    torch_cudnn.deterministic = True # torch.backends.cudnn.deterministic = True
    torch_cudnn.benchmark = False # torch.backends.cudnn.benchmark = False
    torch_use_deterministic_algorithms(True) # torch.use_deterministic_algorithms(True)


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
    class_weights = (1-weight_minority_class, weight_minority_class) 
    adam_betas = tuple(trial.suggest_float(f'adam_beta{i+1}', 0.97, 0.99, step=0.00001) for i in range(2)) if hyperparameters['adam_beta'] is None else hyperparameters['adam_beta']
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, step=0.00001) if hyperparameters['learning_rate'] is None else hyperparameters['learning_rate']
    if "SNN" in MODEL:
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
    else:
        hyperparameters = {
            "name": hyperparameters['name'],
            "topology": hyperparameters['topology'],
            "batch": batch_size,
            "epoch": num_epochs,
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
    # SEEDS
    #fix_seeds(trial.number)
    if MODEL == "FFSNN" or MODEL == "FCSNN":
        model = FFSNN(
            num_features=num_features,
            num_classes=num_classes,
            hyperparameters=hyperparameters,
            gpu_number=int(trial.number)%3,
            verbose=0
        )
    elif MODEL == "CSNN":
        model = CSNN(
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
    else:
        model = ModelSNN(
            num_features=num_features,
            num_classes=num_classes,
            architecture=hyperparameters['name'],
            topology=hyperparameters['topology'],
            class_weights=class_weights,
            betas=betas,
            slope=slope,
            thresholds=thresholds,
            batch_size=batch_size,
            num_epochs=num_epochs,
            num_steps=num_steps,
            adam_betas=adam_betas,
            learning_rate=learning_rate,
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
    experimental_print(f'[{hyperparameters["name"]}] Trial {trial.number}: Recall–{metrics["recall"]*100:.1f}% FPR–{metrics["fpr"]*100:.1f}%') if (metrics["recall"]>0.4 and metrics["fpr"]<0.05) else None
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
    base_path = f"{PATH}/../../data/"
    _, datasets, train_dfs, test_dfs = read_data(base_path, datasets_list, seed=BASE_SEED)
    for dataset_name in datasets.keys(): 
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///dgp-{MACHINE_NAME}-nov.db",
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
    if MACHINE_NAME == "crai01":
        ##SM_1H_K2 = ("SM_1H_K2", [(1, 16, 2), (16, 32, 2), (224, 2)])
        ##SM_1H_K3 = ("SM_1H_K3", [(1, 16, 3), (16, 32, 3), (192, 2)])
        ##SM_1H_K4 = ("SM_1H_K4", [(1, 16, 4), (16, 32, 4), (160, 2)])
        ##SM_1H_K5 = ("SM_1H_K5", [(1, 16, 5), (16, 32, 5), (128, 2)])
        ##MM_1H_K2 = ("MM_1H_K2", [(1, 32, 2), (32, 64, 2), (448, 2)])
        ##MM_1H_K3 = ("MM_1H_K3", [(1, 32, 3), (32, 64, 3), (384, 2)])
        ##MM_1H_K4 = ("MM_1H_K4", [(1, 32, 4), (32, 64, 4), (320, 2)])
        ##MM_1H_K5 = ("MM_1H_K5", [(1, 32, 5), (32, 64, 5), (256, 2)])
        ##LM_1H_K2 = ("LM_1H_K2", [(1, 64, 2), (64, 128, 2), (896, 2)])
        ##LM_1H_K3 = ("LM_1H_K3", [(1, 64, 3), (64, 128, 3), (768, 2)])
        ##LM_1H_K4 = ("LM_1H_K4", [(1, 64, 4), (64, 128, 4), (640, 2)])
        ##LM_1H_K5 = ("LM_1H_K5", [(1, 64, 5), (64, 128, 5), (512, 2)])
        
        FCM_1H = ("FCSNN_1H", (31, 64, 2))
        FCM_2H = ("FCSNN_2H", (31, 64, 64, 2))
        FCM_3H = ("FCSNN_3H", (31, 64, 64, 64, 2))
        topology = FCM_2H

    elif MACHINE_NAME == "crai02":
        ##SM_2H_K2 = ("SM_2H_K2", [(1, 16, 2), (16, 32, 2), (32, 64, 2), (192, 2)])
        ##SM_2H_K3 = ("SM_2H_K3", [(1, 16, 3), (16, 32, 3), (32, 64, 3), (128, 2)])
        ##SM_2H_K4 = ("SM_2H_K4", [(1, 16, 4), (16, 32, 4), (32, 64, 4), (64, 2)])
        ##MM_2H_K2 = ("MM_2H_K2", [(1, 32, 2), (32, 64, 2), (64, 128, 2), (384, 2)])
        ##MM_2H_K3 = ("MM_2H_K3", [(1, 32, 3), (32, 64, 3), (64, 128, 3), (256, 2)])
        ##MM_2H_K4 = ("MM_2H_K4", [(1, 32, 4), (32, 64, 4), (64, 128, 4), (128, 2)])
        ##LM_2H_K2 = ("LM_2H_K2", [(1, 64, 2), (64, 128, 2), (128, 256, 2), (768, 2)])
        ##topology = LM_2H_K2
        #CNN_1H = ("CNN_1H", [(1, 64, 2), (64, 128, 2), (896, 2)])
        #CNN_2H = ("CNN_2H", [(1, 64, 2), (64, 128, 2), (128, 256, 2), (768, 2)])
        #CNN_3H = ("CNN_3H", [(1, 64, 2), (64, 128, 2), (128, 256, 2), (256, 512, 2), (512, 2)])
        #topology = CNN_3H
        FFM_1H = ("FFNN_1H-sigmoid", (31, 64, 2))
        FFM_2H = ("FFNN_2H-sigmoid", (31, 64, 64, 2))
        FFM_3H = ("FFNN_3H-sigmoid", (31, 64, 64, 64, 2))
        topology = FFM_1H
    elif MACHINE_NAME == "crai03":
        ##SM_3H_K2 = ("SM_3H_K2", [(1, 16, 2), (16, 32, 2), (32, 64, 2), (64, 128, 2), (128, 2)])
        ##MM_3H_K2 = ("MM_3H_K2", [(1, 32, 2), (32, 64, 2), (64, 128, 2), (128, 256, 2), (256, 2)])
        ##LM_3H_K2 = ("LM_3H_K2", [(1, 64, 2), (64, 128, 2), (128, 256, 2), (256, 512, 2), (512, 2)])
        ##LM_2H_K4 = ("LM_2H_K4", [(1, 64, 4), (64, 128, 4), (128, 256, 4), (256, 2)])
        ##LM_2H_K3 = ("LM_2H_K3", [(1, 64, 3), (64, 128, 3), (128, 256, 3), (512, 2)])
        ##topology = LM_2H_K4
        pass
    else:
        raise("Unknown machine")
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
    STUDY_NAME = f"20241108-{HYPERPARAMETERS['name']}"
    TRIALS_OPTUNA = 1050
    SAMPLER = TPESampler()
    OBJECTIVE = [("minimize","fpr"), ("maximize","recall")]
    main(DATASETS, STUDY_NAME, TRIALS_OPTUNA, SAMPLER, OBJECTIVE, HYPERPARAMETERS)

