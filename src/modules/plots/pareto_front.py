import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np

def _is_pareto_efficient(costs, maximize_objectives=None):
    """
    Find Pareto-efficient points.
    
    Parameters:
    -----------
    costs : ndarray
        Array with multiple objectives for each point.
    maximize_objectives : list of bool
        Indicates which objectives should be maximized.
        
    Returns:
    --------
    is_efficient : ndarray
        Boolean array indicating Pareto-efficient points.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if maximize_objectives:
            dominance = np.all((costs >= c) if maximize_objectives else (costs <= c), axis=1)
        else:
            dominance = np.all(costs <= c, axis=1)
        is_efficient[i] = ~np.any(dominance & (np.arange(len(costs)) != i))
    return is_efficient

def pareto_front(study, filename="pareto_front.png", path=".", export_eps=False, export_pdf=False, **kwargs):
    """
    Plot the pareto front of a study or a pandas DataFrame.

    Parameters
    ----------
        study : optuna.study.Study or pandas.DataFrame
            Optuna study object or pandas DataFrame.
        filename : str, optional
            Name of the file.
        path : str, optional
            Path to save the plot.
        export_eps: bool, optional
            Boolean to export EPS file.
        export_pdf: bool, optional
            Boolean to export PDF file.
        **kwargs:
            Additional plot parameters.

    Keyword Arguments
    -----------------
        text_scale: float, optional
            Scale of the text (default: 1).
        loc: str, optional
            Location of the legend (default: upper right).
        dpi: int, optional
            Resolution of the plot (default: 1000).
        size: int, optional
            Size of the points (default: 5).
        fpr_col: str, optional
            Column name of the false positive rate (default: 'fpr').
        recall_col: str, optional
            Column name of the recall (default: 'recall').
    """
    extension = filename.split('.')[1]
    filename = filename.split('.')[0]
    text_scale = 1 if not kwargs.get("text_scale") else kwargs.get("text_scale")
    loc = "upper right" if not kwargs.get("loc") else kwargs.get("loc")
    dpi = 1000 if not kwargs.get("dpi") else kwargs.get("dpi")
    size = 5 if not kwargs.get("size") else kwargs.get("size")
    
    # Determine if study is a pandas DataFrame or Optuna study
    if isinstance(study, pd.DataFrame):
        df = study
        fpr_col = kwargs.get('fpr_col','fpr')
        recall_col = kwargs.get('recall_col','recall')
        if 'number' not in df.columns:
            df['number'] = np.arange(len(df))  
        fpr_values = df[fpr_col].values
        recall_values = df[recall_col].values
        costs = np.column_stack([-fpr_values, recall_values])  
        is_best = _is_pareto_efficient(costs, maximize_objectives=[False, True])
        df_best = df[is_best]
        df_not_best = df[~is_best]
    elif study.__class__.__name__ == "Study":
        df = study.trials_dataframe() 
        best_trials = study.best_trials
        fpr_col = kwargs.get('fpr_col','user_attrs_@global fpr')
        recall_col = kwargs.get('recall_col','user_attrs_@global recall')
        df = df.rename(columns={fpr_col: 'fpr', recall_col: 'recall'})
        df_best = df[df['number'].isin([trial.number for trial in best_trials])]
        df_not_best = df[~df['number'].isin([trial.number for trial in best_trials])]
    else:
        raise ValueError("The study must be either an Optuna study object or a pandas DataFrame.")

    # Set seaborn theme and text scale
    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    
    # Set text scale for plot elements
    plt.rc('font', size=plt.rcParams['font.size'] * text_scale)
    plt.rc('axes', titlesize=plt.rcParams['axes.titlesize'] * text_scale)
    plt.rc('axes', labelsize=plt.rcParams['axes.labelsize'] * text_scale)
    plt.rc('xtick', labelsize=plt.rcParams['xtick.labelsize'] * text_scale)
    plt.rc('ytick', labelsize=plt.rcParams['ytick.labelsize'] * text_scale)
    plt.rc('legend', fontsize=plt.rcParams['legend.fontsize'] * text_scale-0.5)
    plt.rc('legend', title_fontsize=plt.rcParams['legend.title_fontsize'] * text_scale-0.5)
    
    # Plot Pareto front
    plt.subplots(figsize=(10, 5))
    plt.scatter(df_not_best['fpr'], df_not_best['recall'], c=df_not_best['number'], cmap='Blues', s=size)
    plt.scatter(df_best['fpr'], df_best['recall'], c=df_best['number'], cmap='Reds', s=size)
    
    # Draw threshold line for FPR = 0.05
    plt.plot([0.05, 0.05], [0,1], 'k--')
    plt.xlim(0, 0.25)
    plt.ylim(0, 1)
    plt.xlabel('FPR')
    plt.ylabel('Recall')
    
    # Add legend
    leg = plt.legend(['Trials', 'Best trials', "5% FPR"], loc=loc)
    leg.legendHandles[0].set_color('darkblue')
    leg.legendHandles[1].set_color('darkred')
    leg.legendHandles[2].set_color('black')
    
    # Save the plot
    if not export_pdf:
        plt.savefig(f'{path}/{filename}.{extension}', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")
    if export_pdf:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)


def pareto_line(studies, filename="pareto_line.png", path=".", export_eps=False, export_pdf=False, **kwargs):
    """
    Plot the pareto front of a study or a pandas DataFrame.

    Parameters
    ----------
        studies : list of optuna.study.Study or pandas.DataFrame
            List of Optuna study objects or pandas DataFrames.
        filename : str, optional
            Name of the file.
        path : str, optional
            Path to save the plot.
        export_eps: bool, optional
            Boolean to export EPS file.
        export_pdf: bool, optional
            Boolean to export PDF file.
        **kwargs:
            Additional plot parameters.

    Keyword Arguments
    -----------------
        text_scale: float, optional
            Scale of the text (default: 1).
        loc: str, optional
            Location of the legend (default: upper right).
        dpi: int, optional
            Resolution of the plot (default: 1000).
        size: int, optional
            Size of the points (default: 5).
        fpr_col: str, optional
            Column name of the false positive rate (default: 'fpr').
        recall_col: str, optional
            Column name of the recall (default: 'recall').
        legend: str, optional
            Title of the legend (default: 'Model').

    """
    extension = filename.split('.')[1]
    filename = filename.split('.')[0]
    text_scale = 1 if not kwargs.get("text_scale") else kwargs.get("text_scale")
    loc = "upper right" if not kwargs.get("loc") else kwargs.get("loc")
    dpi = 1000 if not kwargs.get("dpi") else kwargs.get("dpi")
    size = 5 if not kwargs.get("size") else kwargs.get("size")
    legend = "Model" if not kwargs.get("legend") else kwargs.get("legend")
    
    bests_fronts = {}
    for study in studies:
        if isinstance(study, pd.DataFrame):
            df = study
            fpr_col = kwargs.get('fpr_col','fpr')
            recall_col = kwargs.get('recall_col','recall')
            if 'number' not in df.columns:
                df['number'] = np.arange(len(df))  
            fpr_values = df[fpr_col].values
            recall_values = df[recall_col].values
            costs = np.column_stack([-fpr_values, recall_values])  
            is_best = _is_pareto_efficient(costs, maximize_objectives=[False, True])
            df_best = df[is_best]
            df_not_best = df[~is_best]
        elif study.__class__.__name__ == "Study":
            df = study.trials_dataframe() 
            best_trials = study.best_trials
            fpr_col = kwargs.get('fpr_col','user_attrs_@global fpr')
            recall_col = kwargs.get('recall_col','user_attrs_@global recall')
            df = df.rename(columns={fpr_col: 'fpr', recall_col: 'recall'})
            df_best = df[df['number'].isin([trial.number for trial in best_trials])]
            df_not_best = df[~df['number'].isin([trial.number for trial in best_trials])]
        else:
            raise ValueError("The study must be either an Optuna study object or a pandas DataFrame.")
        bests_fronts[study.study_name] = (df_best, df_not_best)
        

    # Set seaborn theme and text scale
    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    
    # Set text scale for plot elements
    plt.rc('font', size=plt.rcParams['font.size'] * text_scale)
    plt.rc('axes', titlesize=plt.rcParams['axes.titlesize'] * text_scale)
    plt.rc('axes', labelsize=plt.rcParams['axes.labelsize'] * text_scale)
    plt.rc('xtick', labelsize=plt.rcParams['xtick.labelsize'] * text_scale)
    plt.rc('ytick', labelsize=plt.rcParams['ytick.labelsize'] * text_scale)
    plt.rc('legend', fontsize=plt.rcParams['legend.fontsize'] * text_scale-0.5)
    plt.rc('legend', title_fontsize=plt.rcParams['legend.title_fontsize'] * text_scale-0.5)
    
    # Plot Pareto front
    plt.subplots(figsize=(15, 7))#figsize=(3.33, 3.33))
    for study_name, (df_best, df_not_best) in bests_fronts.items():
        sns.lineplot(x='fpr', y='recall', data=df_best, label=study_name)
    
    # Draw threshold line for FPR = 0.05
    plt.plot([0.05, 0.05], [0,1], 'k--')
    plt.xlim(0, 0.25)
    plt.ylim(0, 1)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc=loc, title=legend)
    
    # Save the plot
    if not export_pdf:
        plt.savefig(f'{path}/{filename}.{extension}', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")
    if export_pdf:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)

def pareto_line_grouped(studies_per_config:dict, filename="pareto_line.png", path=".", export_eps=False, export_pdf=False, **kwargs):
    """
    Plot the pareto front of a study or a pandas DataFrame.

    Parameters
    ----------
        studies_per_config : dict of str to list of optuna.study.Study or pandas.DataFrame
            Dictionary with configuration names as keys and lists of Optuna study objects or pandas DataFrames as values.
        filename : str, optional
            Name of the file.
        path : str, optional
            Path to save the plot.
        export_eps: bool, optional
            Boolean to export EPS file.
        export_pdf: bool, optional
            Boolean to export PDF file.
        **kwargs:
            Additional plot parameters.

    Keyword Arguments
    -----------------
        text_scale: float, optional
            Scale of the text (default: 1).
        loc: str, optional
            Location of the legend (default: upper right).
        dpi: int, optional
            Resolution of the plot (default: 1000).
        size: int, optional
            Size of the points (default: 5).
        fpr_col: str, optional
            Column name of the false positive rate (default: 'fpr').
        recall_col: str, optional
            Column name of the recall (default: 'recall').
        legend: str, optional
            Title of the legend (default: 'Model').

    """
    extension = filename.split('.')[1]
    filename = filename.split('.')[0]
    text_scale = 1 if not kwargs.get("text_scale") else kwargs.get("text_scale")
    loc = "upper right" if not kwargs.get("loc") else kwargs.get("loc")
    dpi = 1000 if not kwargs.get("dpi") else kwargs.get("dpi")
    
    # Set seaborn theme and text scale
    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    
    # Set text scale for plot elements
    plt.rc('font', size=plt.rcParams['font.size'] * text_scale)
    plt.rc('axes', titlesize=plt.rcParams['axes.titlesize'] * text_scale)
    plt.rc('axes', labelsize=plt.rcParams['axes.labelsize'] * text_scale)
    plt.rc('xtick', labelsize=plt.rcParams['xtick.labelsize'] * text_scale)
    plt.rc('ytick', labelsize=plt.rcParams['ytick.labelsize'] * text_scale)
    plt.rc('legend', fontsize=plt.rcParams['legend.fontsize'] * text_scale-0.8)
    plt.rc('legend', title_fontsize=plt.rcParams['legend.title_fontsize'] * text_scale-0.8)
    
    fig, ax = plt.subplots(2,3, figsize=(15,7))
    for i, (config, studies) in enumerate(studies_per_config.items()):
        bests_fronts = {}
        for study in studies:
            if isinstance(study, pd.DataFrame):
                df = study
                fpr_col = kwargs.get('fpr_col','fpr')
                recall_col = kwargs.get('recall_col','recall')
                if 'number' not in df.columns:
                    df['number'] = np.arange(len(df))  
                fpr_values = df[fpr_col].values
                recall_values = df[recall_col].values
                costs = np.column_stack([-fpr_values, recall_values])  
                is_best = _is_pareto_efficient(costs, maximize_objectives=[False, True])
                df_best = df[is_best]
                df_not_best = df[~is_best]
            elif study.__class__.__name__ == "Study":
                df = study.trials_dataframe() 
                best_trials = study.best_trials
                fpr_col = kwargs.get('fpr_col','user_attrs_@global fpr')
                recall_col = kwargs.get('recall_col','user_attrs_@global recall')
                df = df.rename(columns={fpr_col: 'fpr', recall_col: 'recall'})
                df_best = df[df['number'].isin([trial.number for trial in best_trials])]
                df_not_best = df[~df['number'].isin([trial.number for trial in best_trials])]
            else:
                raise ValueError("The study must be either an Optuna study object or a pandas DataFrame.")
            bests_fronts[study.study_name] = (df_best, df_not_best)
        
        for study_name, (df_best, df_not_best) in bests_fronts.items():
            sns.lineplot(x='fpr', y='recall', data=df_best, label=study_name, ax=ax[i//3,i%3])
        ax[i//3,i%3].set_title(config)
        # Draw threshold line for FPR = 0.05
        ax[i//3,i%3].plot([0.05, 0.05], [0,1], 'k--')
        ax[i//3,i%3].set_xlim(0, 0.25)
        ax[i//3,i%3].set_ylim(0, 1)
        ax[i//3,i%3].set_xlabel('FPR')
        ax[i//3,i%3].set_ylabel('TPR')
        ax[i//3,i%3].legend(loc=loc, title="Model", ncol=2 if len(bests_fronts)>10 else 1)
    plt.tight_layout()
    # Save the plot
    if not export_pdf:
        plt.savefig(f'{path}/{filename}.{extension}', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")
    if export_pdf:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)