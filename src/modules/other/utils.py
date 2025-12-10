
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from optuna.samplers import RandomSampler
from optuna.trial import FixedTrial
from optuna import distributions

def read_data(base_path, datasets_list=["Base"], seed=None):
    datasets_paths = {
        key : f"{base_path}{key}.parquet" if key == "Base" else f"{base_path}{key.split()[-1]}.parquet" for key in datasets_list
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