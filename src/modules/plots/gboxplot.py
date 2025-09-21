import matplotlib.pyplot as plt
import seaborn as sns
import os

def gboxplot(df, filename="gboxplot.pdf", path=".", export_eps=False, export_pdf=False, **kwargs):
    """
    Plot a syringe plot, a boxplot with jittered points

    Parameters
    ----------
        df : pd.DataFrame
            list of DataFrames
        filename : str, optional
            name of the file
        path : str, optional
            path to save the plot
        export_eps: bool, optional
            boolean to export eps file
        export_pdf: bool, optional
            boolean to export pdf file
        **kwargs:
            see below

    Keyword Arguments
    ----------
        figsize: tuple, optional
            size of the figure (default: (10, 5))
        x: str, optional
            x column name (default: metric)
        y: str, optional
            y column name (default: percentage)
        z: str, optional 
            z column name (default: model)
        x_name: str, optional 
            x label (default: Metric)
        y_name: str, optional
            y label (default: "Percentage")
        z_name: str, optional
            z label (default: Model)
        text_scale: float, optional
            scale of the text (default: 1)
        loc: str, optional
            location of the legend (default: upper right)
        ncol: int, optional
            number of columns of the legend (default: 1)
        bbox_to_anchor: tuple, optional
            position of the legend (default: None)
        y_scale: bool, optional
            log scale of the y axis (default: False)
        dpi: int, optional
            dpi of the plot (default, 1000)

    """
    figsize = kwargs.get("figsize", (10, 5))
    x = kwargs.get("x", "metric")
    y = kwargs.get("y", "value")
    z = kwargs.get("z", "model")
    x_name = kwargs.get("x_name", "Metric")
    y_name = kwargs.get("y_name", "Value")
    z_name = kwargs.get("z_name", "Model")
    text_scale = kwargs.get("text_scale", 1)
    loc = kwargs.get("loc", "upper right")
    ncol = kwargs.get("ncol", 1)
    bbox_to_anchor = kwargs.get("bbox_to_anchor", None)
    y_scale = kwargs.get("y_scale", None)
    dpi = kwargs.get("dpi", 1000)
    extension = filename.split('.')[1]
    filename = filename.split('.')[0]
    # violin plot grouped by network, epoch and batch size
    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    # set text scale of axes, ticks and labels
    # get current font size
    plt.rc('font', size=plt.rcParams['font.size'] * text_scale)
    plt.rc('axes', titlesize=plt.rcParams['axes.titlesize'] * text_scale)
    plt.rc('axes', labelsize=plt.rcParams['axes.labelsize'] * text_scale)
    plt.rc('xtick', labelsize=plt.rcParams['xtick.labelsize'] * text_scale)
    plt.rc('ytick', labelsize=plt.rcParams['ytick.labelsize'] * text_scale)
    plt.rc('legend', fontsize=plt.rcParams['legend.fontsize'] * text_scale-0.8)
    plt.rc('legend', title_fontsize=plt.rcParams['legend.title_fontsize'] * text_scale-0.8)

    _, ax = plt.subplots(1, 1, figsize=figsize)
    sns.boxplot(x=x, y=y, hue=z, data=df, fliersize=2, palette="tab10")
    # only show legend for the first plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title=z_name, loc=loc, ncol=ncol, bbox_to_anchor=bbox_to_anchor)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    # set log scale
    if y_scale:
        ax.set_yscale('log')
    plt.tight_layout()
    if not export_pdf:
        plt.savefig(f"{path}/{filename}.{extension}", format=extension, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")
    if export_pdf:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
