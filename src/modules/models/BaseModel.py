import torch

from torch import from_numpy
from torch.utils.data import DataLoader
from modules.dataset.DatasetBAF import DatasetBAF
from modules.metrics import evaluate, evaluate_business_constraint, evaluate_fairness

class BaseModel(object):
    def __init__(self, num_features, num_classes, hyperparameters, **kwargs):
        super().__init__()
        self.verbose = kwargs.get('verbose', 0)
        self.gpu_number = kwargs.get('gpu_number', 0)
        device = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                device = torch.device(f"cuda:{self.gpu_number}")
                print(f"Selected GPU {self.gpu_number}: {device}")
            except Exception:
                device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS.")
        else:
            print("No GPU available, using CPU.")
        self.num_features = num_features
        self.num_classes = num_classes
        self.topology = hyperparameters['topology']
        self.hyperparameters = hyperparameters
        self.dtype = torch.float
        self.device = device
    
    def load_data(self, x, y):
        """
        DEPRECATED
        """
        x_np = from_numpy(x.values).float().unsqueeze(1)
        y_np = from_numpy(y.values).int()
        ds = DatasetBAF(x_np, y_np)
        loader = DataLoader(ds, batch_size=self.hyperparameters['batch'], shuffle=True, drop_last=False, num_workers=0)
        return loader

    def evaluate(self, targets, predicted):
        """Evaluate the model using the confusion matrix and some metrics.
        ------------------------------------------------------ 
        Args:
            targets (list): list of true values
            predicted (list): list of predicted values
        ------------------------------------------------------ 
        Returns:
            cm (np.array): confusion matrix
            accuracy (float): accuracy of the model
            precision (float): precision of the model
            recall (float): recall of the model
            fpr (float): false positive rate of the model
            f1_score (float): f1 score of the model
            auc (float): area under the curve of the model
        """
        metrics = evaluate(targets, predicted)
        print_str = [
            "=======================================",
            'Confusion Matrix:',
            f"{metrics['tp']}(TP)\t{metrics['fn']}(FN)",
            f"{metrics['fp']}(FP)\t{metrics['tn']}(TN)",
            "---------------------------------------",
            f'FPR:\t\t{metrics["fpr"]*100:.4f}%',
            f'Recall:\t\t{metrics["recall"]*100:.4f}%',
            f'TNR:\t\t{metrics["tnr"]*100:.4f}%',
            f'Accuracy:\t{metrics["accuracy"]*100:.4f}%',
            f'Precision:\t{metrics["precision"]*100:.4f}%',
            f'F1 Score:\t{metrics["f1_score"]*100:.4f}%',
            f'AUC:\t\t{metrics["auc"]*100:.4f}%',
            "=======================================",
        ]
        print("\n".join(print_str)) if self.verbose >= 1 and (metrics["recall"]>0.1 and metrics["fpr"]<0.1) else None
        return metrics
    
    def evaluate_business_constraint(self, y_test, predictions):
        """Evaluate the model using the business constraint of 5% FPR.
        ------------------------------------------------------
        Args:
            y_test (pd.Series): series with the test labels
            predictions (np.array): array with the predictions
        ------------------------------------------------------
        Returns:
            threshold (float): threshold for the model
            fpr@5FPR (float): false positive rate of the model
            recall@5FPR (float): recall of the model
            tnr@5FPR (float): true negative rate of the model
            accuracy@5FPR (float): accuracy of the model
            precision@5FPR (float): precision of the model
            f1_score@5FPR (float): f1 score of the model
        """
        metrics = evaluate_business_constraint(y_test, predictions)
        print_str = [
            "=======================================",
            '5% FPR Metrics:',
            '---------------------------------------',
            f'FPR@5FPR:\t{metrics["fpr@5FPR"]*100:.4f}%',
            f'Recall@5FPR:\t{metrics["recall@5FPR"]*100:.4f}%',
            f'TNR@5FPR:\t{metrics["tnr@5FPR"]*100:.4f}%',
            f'Accuracy@5FPR:\t{metrics["accuracy@5FPR"]*100:.4f}%',
            f'Precision@5FPR:\t{metrics["precision@5FPR"]*100:.4f}%',
            f'F1-Score@5FPR:\t{metrics["f1_score@5FPR"]*100:.4f}%',
            "=======================================",
        ]
        print("\n".join(print_str)) if self.verbose >= 1 and (metrics["recall@5FPR"]>0.1 and metrics["fpr@5FPR"]<0.1) else None
        return metrics
    
    def evaluate_fairness(self, x_test, y_test, predictions, sensitive_attribute, attribute_threshold):
        """Evaluate the model using the Aequitas library.
        ------------------------------------------------------
        Args:
            x_test (pd.DataFrame): dataframe with the test features
            y_test (pd.Series): series with the test labels
            predictions (np.array): array with the predictions
            sensitive_attribute (str): name of the sensitive attribute
            attribute_threshold (float): threshold for the sensitive attribute
        ------------------------------------------------------
        Returns:
            fpr_ratio (float): false positive rate ratio of the model
            fnr_ratio (float): false negative rate ratio of the model
            recall_older (float): recall of the older group
            recall_younger (float): recall of the younger group
            fpr_older (float): false positive rate of the older group
            fpr_younger (float): false positive rate of the younger group
            fnr_older (float): false negative rate of the older group
            fnr_younger (float): false negative rate of the younger group
        """
        metrics = evaluate_fairness(x_test, y_test, predictions, sensitive_attribute, attribute_threshold)
        print_str = [
            "=======================================",
            f'Fairness Metrics for {sensitive_attribute} > {attribute_threshold}:',
            '---------------------------------------',
            f'FPR Ratio:\t{metrics["fpr_ratio"]*100:.4f}',
            f'FNR Ratio:\t{metrics["fnr_ratio"]*100:.4f}',
            "=======================================",
        ]
        print("\n".join(print_str)) if self.verbose >= 1 and (metrics["recall_older"]>0.1 and metrics["fpr_older"]<0.1) else None
        return metrics
    
    def get_parameters(self):
        return [p for p in self.network.parameters() if p.requires_grad]
    
    def get_num_params(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)