import seaborn as sns   
import numpy as np
import os

from copy import deepcopy
from matplotlib import pyplot as plt   



def performance_fairness(df, metrics, filename="performance_fairness.png", path=".", export_eps=False, export_pdf=False, **kwargs):
    """
    Plot performance vs fairness
    :param df: DataFrame with columns model, dataset, performance, fairness
    :param metrics: list of metrics to plot
    :param top_n: number of top models to highlight
    :param best_models: boolean to highlight best models
    :param filename: name of the file to save the plot
    :param path: path to save the plot
    :param export_eps: boolean to export eps file
    :param kwargs: x_name, y_name, z_name
    """
    # set nan values to 0
    x_name="xlabel" if not kwargs.get("x_name") else kwargs.get("x_name")
    y_name="ylabel" if not kwargs.get("y_name") else kwargs.get("y_name")
    z_name="zlabel" if not kwargs.get("hue_name") else kwargs.get("hue_name")
    text_scale = 1 if not kwargs.get("text_scale") else kwargs.get("text_scale")
    alpha = 1 if not kwargs.get("alpha") else kwargs.get("alpha")
    size = 50 if not kwargs.get("size") else kwargs.get("size")
    loc = "upper right" if not kwargs.get("loc") else kwargs.get("loc")
    dpi = 1000 if not kwargs.get("dpi") else kwargs.get("dpi")
    extension = filename.split('.')[1]
    filename = filename.split('.')[0]
    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    plt.figure(figsize=(5, 5))
    plt.rc('font', size=plt.rcParams['font.size'] * text_scale)
    plt.rc('axes', titlesize=plt.rcParams['axes.titlesize'] * text_scale)
    plt.rc('axes', labelsize=plt.rcParams['axes.labelsize'] * text_scale)
    plt.rc('xtick', labelsize=plt.rcParams['xtick.labelsize'] * text_scale)
    plt.rc('ytick', labelsize=plt.rcParams['ytick.labelsize'] * text_scale)
    plt.rc('legend', fontsize=plt.rcParams['legend.fontsize'] * text_scale-0.5)
    plt.rc('legend', title_fontsize=plt.rcParams['legend.title_fontsize'] * text_scale-0.5)
    df = df.dropna()
    # change alpha if is_top
    sns.jointplot(data=df, x=metrics[0], y=metrics[1], hue=metrics[2], alpha=alpha, joint_kws={'s': size})
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.legend(title=z_name, loc=loc)
    if not export_pdf:
        plt.savefig(f'{path}/{filename}.{extension}', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")
    if export_pdf:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        