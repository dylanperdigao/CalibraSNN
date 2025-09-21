import seaborn as sns   
import os

from matplotlib import pyplot as plt   



def annoted_scatter(df, filename="annoted_scatter.png", path=".", export_eps=False, export_pdf=False, **kwargs):
    """
    Plot an annoted scatter plot with the given DataFrame
    :param df: DataFrame with columns model, dataset, performance, fairness
    :param filename: name of the file to save the plot
    :param path: path to save the plot
    :param export_eps: boolean to export eps file
    :param kwargs: x_name, y_name, z_name
    """
    # set nan values to 0
    x_name="xlabel" if not kwargs.get("x_name") else kwargs.get("x_name")
    y_name="ylabel" if not kwargs.get("y_name") else kwargs.get("y_name")
    z_name=None if not kwargs.get("z_name") else kwargs.get("z_name")
    xlog = False if not kwargs.get("xlog") else kwargs.get("xlog")
    ylog = False if not kwargs.get("ylog") else kwargs.get("ylog")
    text_scale = 1 if not kwargs.get("text_scale") else kwargs.get("text_scale")
    size = 50 if not kwargs.get("size") else kwargs.get("size")
    loc = "upper right" if not kwargs.get("loc") else kwargs.get("loc")
    dpi = 1000 if not kwargs.get("dpi") else kwargs.get("dpi")
    fig_size = (10, 7) if not kwargs.get("fig_size") else kwargs.get("fig_size")
    fig_theme = "darkgrid" if not kwargs.get("fig_theme") else kwargs.get("fig_theme")
    extension = filename.split('.')[1]
    filename = filename.split('.')[0]
    sns.set_theme(style=fig_theme)
    #sns.set_context("paper")
    plt.figure(figsize=fig_size)
    plt.rc('font', size=plt.rcParams['font.size'] * text_scale)
    plt.rc('axes', titlesize=plt.rcParams['axes.titlesize'] * text_scale)
    plt.rc('axes', labelsize=plt.rcParams['axes.labelsize'] * text_scale)
    plt.rc('xtick', labelsize=plt.rcParams['xtick.labelsize'] * text_scale)
    plt.rc('ytick', labelsize=plt.rcParams['ytick.labelsize'] * text_scale)
    plt.rc('legend', fontsize=plt.rcParams['legend.fontsize'] * text_scale-0.5)
    plt.rc('legend', title_fontsize=plt.rcParams['legend.title_fontsize'] * text_scale-0.5)
    # change alpha if is_top
    sns.scatterplot(
        data=df,
        x=2,
        y=1,
        hue=0,
        s=size
    )
    # Annotate points with model names
    for i, row in df.iterrows():
        plt.text(row[2],
            row[1]-5, 
            row[0], 
            fontsize=10*text_scale,
            ha="right",
        )
    plt.ylim([0, 105])
    if xlog:
        plt.xscale("log")
    if ylog:
        plt.yscale("log")
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.legend(title=z_name, loc=loc) if z_name else plt.legend().remove()
    if not export_pdf:
        plt.savefig(f'{path}/{filename}.{extension}', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")
    if export_pdf:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
