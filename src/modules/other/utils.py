import os
import pandas as pd
import paho.mqtt.publish as publish

from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from torch import Generator
from modules.dataset import BAF, Iris, UNSW_NB15
from optuna.samplers import RandomSampler
from optuna.trial import FixedTrial
from optuna import distributions

# MQTT broker configuration
MQTT_BROKER = 'broker.hivemq.com'
MQTT_PORT = 1883
MQTT_TOPIC = '/dylanperdigao/status' 

def experimental_print(message):
    print(message)
    try:
        if 'crai' in os.uname().nodename:
            message = f"[{os.uname().nodename}]{message}"
        else:
            message = f"[local]{message}"
        publish.single(MQTT_TOPIC, message, hostname=MQTT_BROKER, port=MQTT_PORT)
    except Exception:
        pass

def read_data(base_path, datasets_list=["Base"], seed=None):
    """
    DEPRECATED
    Read the datasets from the parquet files and return the train and test sets.
    """
    datasets_paths = {
        key : f"{base_path}{key}.parquet" if key == "Base" else f"{base_path}Variant {key.split()[-1]}.parquet" for key in datasets_list
    }
    # Read the datasets with pandas.
    datasets = {key: pd.read_parquet(path) for key, path in datasets_paths.items()}
    categorical_features = [
        "payment_type",
        "employment_status",
        "housing_status",
        "source",
        "device_os",
    ]
    train_dfs = {key: df[df["month"]<6].sample(frac=1, replace=False, random_state=seed) for key, df in datasets.items()}
    test_dfs = {key: df[df["month"]>=6].sample(frac=1, replace=False, random_state=seed)  for key, df in datasets.items()}
    for name in datasets.keys(): 
        train = train_dfs[name]
        test = test_dfs[name]
        for feat in categorical_features:
            encoder = LabelEncoder()
            encoder.fit(train[feat]) 
            train[feat] = encoder.transform(train[feat]) 
            test[feat] = encoder.transform(test[feat]) 
    return datasets_paths, datasets, train_dfs, test_dfs

def sample_data(df, n, seed=None, verbose=False):
    """Sample data preserving class distribution.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing data to be sampled.
    n : int
        Number of samples to be drawn.

    Returns
    -------
    df_sample : pandas.DataFrame
        Dataframe containing sampled data.
    """
    if n > len(df):
        print("n is greater than the number of samples")
        return df
    df_sample = pd.DataFrame()
    counter = Counter(df.iloc[:, -1])
    total = sum(counter.values())
    proportion = {key: value / total for key, value in counter.items()}
    print(proportion) if verbose else None
    for key, value in proportion.items():
        df_class = df[df.iloc[:, -1] == key]
        samples = round(n * value)
        if samples <= 10:
            print(f"Class {key} has less than 10 samples.") if verbose else None
            df_class_sample = df_class.sample(n=10, random_state=seed, replace=True)
        else:
            df_class_sample = df_class.sample(n=samples, random_state=seed)
        df_sample = pd.concat([df_sample, df_class_sample])
    df_sample = df_sample.sort_index()
    return df_sample

def smote_rebalance(df, percentage=0.3, seed=None, return_df=True, verbose=False):
    """
    Rebalance data using SMOTE
    :param df: dataframe
    :param percentage: percentage of minority class
    :return: rebalanced dataframe
    """
    df_copy = df.copy()
    # split data
    x = df_copy.drop(["label"], axis=1)
    y = df_copy["label"]
    print("Oversampling with SMOTE") if verbose else None
    # get majority class
    counter = Counter(y.to_numpy().tolist())
    majority_class_label = max(counter, key=counter.get)
    # dict with 30% more samples for each class except the majority class
    sampling_strategy = {k: int(v * (1+percentage)) for k, v in counter.items() if k != majority_class_label}  
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=seed)
    x_smote, y_smote = smote.fit_resample(x, y)
    # create dataframe
    if return_df:
        df_smote = pd.DataFrame(x_smote, columns=df_copy.columns[:-1])
        df_smote["label"] = y_smote
        print("Oversampling with SMOTE done") if verbose else None
        return df_smote
    return x_smote.to_numpy(), y_smote.to_numpy()


def load_dataset(dataset: str, batch_size: int, root='./data', seed=None):
    """
    Load the dataset and return the train and test loaders.\n
    ---
    DATASETS
    ---

    BAF - Bank Account Fraud:
    - Variant: Base, TypeI, TypeII, TypeIII, TypeIV, TypeV
    - Dataset size: 1,000,000
    - Features: 31 or 33
    - Classes: 2

    MNIST:
    - Dataset size: 60,000 / 10,000
    - Image size: 28 x 28
    - Classes: 10
    
    Fashion MNIST:
    - Dataset size: 60,000 / 10,000
    - Image size: 28 x 28
    - Classes: 10

    CIFAR10:
    - Dataset size: 50,000 / 10,000
    - Image size: 32 x 32
    - Classes: 10

    CIFAR100:
    - Dataset size: 50,000 / 10,000
    - Image size: 32 x 32
    - Classes: 100

    UNSW-NB15:
    - Dataset size: 175,341
    - Features: 49
    - Classes: 10

    IRIS:
    - Dataset size: 150
    - Features: 4
    - Classes: 3

    ---
    INPUTS/OUTPUTS
    ---

    Args:
        dataset (str): Name of the dataset (baf-{variant}, mnist, fashion_mnist, cifar10, cifar100).
        batch_size (int): Batch size.
        root (str): Path to the dataset.
        seed (int): Seed for the random

    Returns:
        DataLoader: Train data loader.
        DataLoader: Test data loader.
    """
    transform_mnist = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
    ])
    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if dataset == 'baf':
        train = BAF(variant='Base', root=f"{root}/BAF", train=True)
        test = BAF(variant='Base', root=f"{root}/BAF", train=False)
    elif 'baf' in dataset.lower():
        train = BAF(variant=dataset.split('-')[-1], root=f"{root}/BAF", train=True)
        test = BAF(variant=dataset.split('-')[-1], root=f"{root}/BAF", train=False)
    elif dataset == 'mnist':
        train = MNIST(root=root, train=True, download=True, transform=transform_mnist)
        test = MNIST(root=root, train=False, download=True, transform=transform_mnist)
    elif dataset == 'fashion_mnist':
        train = FashionMNIST(root=root, train=True, download=True, transform=transform_mnist)
        test = FashionMNIST(root=root, train=False, download=True, transform=transform_mnist)
    elif dataset == 'cifar10':
        train = CIFAR10(root=root, train=True, download=True, transform=transform_cifar)
        test = CIFAR10(root=root, train=False, download=True, transform=transform_cifar)
    elif dataset == 'cifar100':
        train = CIFAR100(root=root, train=True, download=True, transform=transform_cifar)
        test = CIFAR100(root=root, train=False, download=True, transform=transform_cifar)
    elif dataset == 'unsw_nb15':
        train = UNSW_NB15(root=f"{root}/UNSW-NB15", train=True, multiclass=True)
        test = UNSW_NB15(root=f"{root}/UNSW-NB15", train=False, multiclass=True)
    elif dataset == 'iris':
        train = Iris(train=True)
        test = Iris(train=False)
    else:
        raise ValueError("Invalid dataset")
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, generator=Generator().manual_seed(seed))
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=0, generator=Generator().manual_seed(seed))
    return train_loader, test_loader

class RandomTrial(FixedTrial):
    """
    Class to create a random trial
    ----------
    Parameters
    ----------
        seed : int
            seed for the random generator
        sampler : optuna.samplers.BaseSampler
            sampler for the random generator
        number : int, optional
            number of the trial
    """
    def __init__(self, seed=None, sampler=None, number=0):
        super().__init__(params=None, number=number)
        self.seed = seed
        self.sampler = sampler or RandomSampler(self.seed)

    def _suggest(self, name, distribution):
        if name in self._distributions:
            distributions.check_distribution_compatibility(self._distributions[name], distribution)
            param_value = self._suggested_params[name]
        elif self._is_fixed_param(name, distribution):
            param_value = self.system_attrs["fixed_params"][name]
        elif distribution.single():
            param_value = distributions._get_single_value(distribution)
        else:
            param_value = self.sampler.sample_independent(study=None, trial=self, param_name=name, param_distribution=distribution)
        self._suggested_params[name] = param_value
        self._distributions[name] = distribution
        return self._suggested_params[name]
    
    def _is_fixed_param(self, name, distribution):
        system_attrs = self._system_attrs
        if "fixed_params" not in system_attrs:
            return False
        if name not in system_attrs["fixed_params"]:
            return False
        param_value = system_attrs["fixed_params"][name]
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)
        contained = distribution._contains(param_value_in_internal_repr)
        return contained