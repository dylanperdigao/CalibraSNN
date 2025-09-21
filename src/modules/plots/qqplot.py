import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import os

def qqplot(df, filename="qqplot.pdf", path=".", export_eps=False, export_pdf=False, **kwargs):
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
        title: str, optional
            title of the plot (default: None)
        titles: list, optional
            list of titles for each plot (default: None)
        text_scale: float, optional
            scale of the text (default: 1)
        dpi: int, optional
            dpi of the plot (default, 1000)

    """
    figsize = kwargs.get("figsize", (10, 10))
    titles = kwargs.get("titles", None) 
    title = kwargs.get("title", None)
    text_scale = kwargs.get("text_scale", 1)
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

    fig, ax = plt.subplots(1, 4, figsize=figsize, sharex=True,sharey=True, layout='constrained')
    for i, data in enumerate(df.columns[1:]):
        if len(ax.shape) > 1:
            st.probplot(df[data], dist="norm", plot=ax[i//2, i%2])
            if titles is not None:
                ax[i//2, i%2].set_title(f'{titles[i]} QQ-plot') 
            ax[i//2, i%2].set_xlabel(None) 
            ax[i//2, i%2].set_ylabel(None) 
        else:
            st.probplot(df[data], dist="norm", plot=ax[i])
            if titles is not None:
                ax[i].set_title(f'{titles[i]} QQ-plot') 
            ax[i].set_xlabel(None) 
            ax[i].set_ylabel(None)
    fig.supxlabel('Theoretical quantiles')
    fig.supylabel('Ordered Values')
    if title is not None:
        fig.suptitle(title)
    #fig.tight_layout(h_pad=5, w_pad=-5)
    if not export_pdf:
        plt.savefig(f"{path}/{filename}.{extension}", format=extension, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")
    if export_pdf:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)