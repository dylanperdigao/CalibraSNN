import matplotlib.pyplot as plt
import seaborn as sns
import os

def syringeplot(df, filename="syringeplot.pdf", path=".", export_eps=False, export_pdf=False, **kwargs):
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
        x: str, optional
            x column name (default: config)
        y: str, optional
            y column name (default: recall)
        z: str, optional 
            z column name (default: net)
        x_name: str, optional 
            x label (default: Configuration)
        y_name: str, optional
            y label (default: Recall)
        z_name: str, optional
            z label (default: Network)
        text_scale: float, optional
            scale of the text (default: 1)
        loc: str, optional
            location of the legend (default: upper right)
        dpi: int, optional
            dpi of the plot (default, 1000)

    """
    ax = kwargs.get("ax", None)
    x = kwargs.get("x", "config")
    y = kwargs.get("y", "recall")
    z = kwargs.get("z", "net")
    x_name = kwargs.get("x_name", "Configuration")
    y_name = kwargs.get("y_name", "Recall")
    z_name = kwargs.get("z_name", "Network")
    text_scale = kwargs.get("text_scale", 1)
    loc = kwargs.get("loc", "upper right")
    dpi = kwargs.get("dpi", 1000)
    palette = kwargs.get("palette", None)
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
    plt.rc('legend', fontsize=(plt.rcParams['legend.fontsize'] * text_scale)-3*text_scale)
    plt.rc('legend', title_fontsize=(plt.rcParams['legend.title_fontsize'] * text_scale)-3*text_scale)
    _, ax = plt.subplots(1, 1, figsize=(10, 5)) if ax is None else (None, ax)
    sns.boxplot(data=df, x=x, y=y, hue=z, ax=ax, gap=0.1, boxprops=dict(alpha=0.3), fliersize=1, palette=palette)
    sns.stripplot(data=df, x=x, y=y, hue=z, ax=ax, dodge=True, jitter=True, alpha=1, size=1, palette=palette)
    # only show legend for the first plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:int(len(labels)/2)], labels[:int(len(labels)/2)], title=z_name, loc=loc)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    if not export_pdf:
        plt.savefig(f"{path}/{filename}.{extension}", format=extension, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")
    if export_pdf:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
