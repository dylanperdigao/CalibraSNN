import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def tradeoff_performance_fairness(df, performance_metric="performance", fairness_metric="fairness", filename="tradeoff_performance_fairness.png", path=".", export_eps=False, export_pdf=False, **kwargs):
    """
    Plot tradeoff between performance and fairness
    :param df: DataFrame with columns model, dataset, performance, fairness
    :param performance_metric: performance metric
    :param fairness_metric: fairness metric
    :param path: path to save the plot
    :param kwargs: x_label, y_label, title
    """
    loc = "upper right" if not kwargs.get("loc") else kwargs.get("loc")
    text_scale = 1 if not kwargs.get("text_scale") else kwargs.get("text_scale")
    fig_size = (3.33, 3.33) if not kwargs.get("fig_size") else kwargs.get("fig_size")
    extension = filename.split('.')[1]
    filename = filename.split('.')[0]
    dpi = 1000 if not kwargs.get("dpi") else kwargs.get("dpi")
    sns.set_style("darkgrid")
    sns.set_context("paper") 
    plt.rc('font', size=plt.rcParams['font.size'] * text_scale)
    plt.rc('axes', titlesize=plt.rcParams['axes.titlesize'] * text_scale)
    plt.rc('axes', labelsize=plt.rcParams['axes.labelsize'] * text_scale)
    plt.rc('xtick', labelsize=plt.rcParams['xtick.labelsize'] * text_scale)
    plt.rc('ytick', labelsize=plt.rcParams['ytick.labelsize'] * text_scale)
    plt.rc('legend', fontsize=plt.rcParams['legend.fontsize'] * text_scale-0.5)
    plt.rc('legend', title_fontsize=plt.rcParams['legend.title_fontsize'] * text_scale-0.5)
    plt.figure(figsize=fig_size)
    tradeoff_list = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        for alpha in range(0, 101, 1):
            alpha = alpha / 100
            tradeoff = alpha * model_df[performance_metric].mean() + (1 - alpha) * model_df[fairness_metric].mean()
            std_dev = ((alpha * model_df[performance_metric]).std()**2 + ((1 - alpha) * model_df[fairness_metric]).std()**2)**0.5
            tradeoff_list.append({"model": model, "alpha": alpha, "tradeoff": tradeoff, "std_dev": std_dev})
    # Convert list to DataFrame
    tradeoff_df = pd.DataFrame(tradeoff_list)
    # Plot using seaborn lineplot
    sns.lineplot(data=tradeoff_df, x="alpha", y="tradeoff", hue="model")
    for name, group in tradeoff_df.groupby("model"):
        plt.fill_between(group['alpha'], group['tradeoff'] - group['std_dev'], group['tradeoff'] + group['std_dev'], alpha=0.1)
    if loc == "top":
        plt.legend(loc="lower left", bbox_to_anchor=(-0.05, 1), ncol=4, title=None, frameon=False)
    else:
        plt.legend(title="Model", loc=loc)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(kwargs.get("x_label")) if kwargs.get("x_label") else plt.xlabel("\u03B1")
    plt.ylabel(kwargs.get("y_label")) if kwargs.get("y_label") else plt.ylabel(f"\u03B1 $\cdot$ {performance_metric} + (1-\u03B1) $\cdot$ {fairness_metric}")
    plt.title(kwargs.get("title")) if kwargs.get("title") else None
    if not export_pdf:
        plt.savefig(f'{path}/{filename}.{extension}', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")
    if export_pdf:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)


def tradeoff_performance_fairness_grouped(dfs, performance_metric="performance", fairness_metric="fairness", filename="tradeoff_performance_fairness.png", path=".", export_eps=False, export_pdf=False, **kwargs):
    """
    Plot tradeoff between performance and fairness
    :param df: list of DataFrames with columns model, dataset, performance, fairness
    :param performance_metric: performance metric
    :param fairness_metric: fairness metric
    :param path: path to save the plot
    :param kwargs: x_label, y_label, title
    """
    loc = "upper right" if not kwargs.get("loc") else kwargs.get("loc")
    text_scale = 1 if not kwargs.get("text_scale") else kwargs.get("text_scale")
    fig_size = (3.33, 3.33) if not kwargs.get("fig_size") else kwargs.get("fig_size")
    extension = filename.split('.')[1]
    filename = filename.split('.')[0]
    dpi = 1000 if not kwargs.get("dpi") else kwargs.get("dpi")
    sns.set_style("darkgrid")
    sns.set_context("paper") 
    plt.rc('font', size=plt.rcParams['font.size'] * text_scale)
    plt.rc('axes', titlesize=plt.rcParams['axes.titlesize'] * text_scale)
    plt.rc('axes', labelsize=plt.rcParams['axes.labelsize'] * text_scale)
    plt.rc('xtick', labelsize=plt.rcParams['xtick.labelsize'] * text_scale)
    plt.rc('ytick', labelsize=plt.rcParams['ytick.labelsize'] * text_scale)
    plt.rc('legend', fontsize=plt.rcParams['legend.fontsize'] * text_scale * 0.75)
    plt.rc('legend', title_fontsize=plt.rcParams['legend.title_fontsize'] * text_scale * 0.75)
    
    fig, ax = plt.subplots(1, 3, figsize=fig_size)
    # columns that contain fairnss_metric
    for i, (df,attr) in enumerate(dfs):
        tradeoff_list = []
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            for alpha in range(0, 101, 1):
                alpha = alpha / 100
                tradeoff = alpha * model_df[performance_metric].mean() + (1 - alpha) * model_df[f'{fairness_metric}_{attr}'].mean()
                std_dev = ((alpha * model_df[performance_metric]).std()**2 + ((1 - alpha) * model_df[f'{fairness_metric}_{attr}']).std()**2)**0.5
                tradeoff_list.append({"model": model, "alpha": alpha, "tradeoff": tradeoff, "std_dev": std_dev})
        # Convert list to DataFrame
        tradeoff_df = pd.DataFrame(tradeoff_list)
        # Plot using seaborn lineplot
        sns.lineplot(data=tradeoff_df, x="alpha", y="tradeoff", hue="model", ax=ax[i])
        for name, group in tradeoff_df.groupby("model"):
            ax[i].fill_between(group['alpha'], group['tradeoff'] - group['std_dev'], group['tradeoff'] + group['std_dev'], alpha=0.1)
        if loc == "top":
            ax[i].legend(loc="lower left", bbox_to_anchor=(-0.05, 1), ncol=4, title=None, frameon=False)
        else:
            ax[i].legend(title="Model", loc=loc)
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0, 1)
        ax[i].set_xlabel(kwargs.get("x_label")) if kwargs.get("x_label") else ax[i].set_xlabel("\u03B1")
        ax[i].set_ylabel(kwargs.get("y_label")) if kwargs.get("y_label") else ax[i].set_ylabel(f"\u03B1 $\cdot$ {performance_metric} + (1-\u03B1) $\cdot$ {fairness_metric}")
        ax[i].set_title(attr.capitalize())
    plt.tight_layout()
    if not export_pdf:
        plt.savefig(f'{path}/{filename}.{extension}', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")
    if export_pdf:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)