import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler, MinMaxScaler

PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{PATH}/../")
from modules.other.utils import sample_data

def correlation(datasets, path='.', export_eps=False):
    """
    Correlation matrix

    Arguments:
    - datasets {dict} -- Dictionary with datasets
    - path {str} -- Path to save the plots
    """
    # Device fraud count está sempre a 0, logo n aparece na heatmap
    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    for name, dataset in datasets.items():
        filename = f"corr-{name}"
        corr = dataset.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        _, ax = plt.subplots(figsize=(10, 10))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        plt.savefig(f'{path}/{filename}.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
        if export_eps:
            plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
            os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
            os.system(f"rm {path}/{filename}.pdf")
        plt.close()

def class_distribution(datasets, path='.', export_eps=False):
    """
    Class distribution
    
    Arguments:
    - datasets {dict} -- Dictionary with datasets
    - path {str} -- Path to save the plots
    """
    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    for name, dataset in datasets.items():
        filename = f"class-{name}"
        plt.figure(figsize=(10, 10))
        sns.countplot(x='fraud_bool', data=dataset)
        # show percentages above bars
        ncount = len(dataset)
        ax = plt.gca()
        ax.set_ylim(0, ncount)
        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate(f'{100. * y / ncount:.1f}%\n({y:.0f})', (x.mean(), y), ha='center', va='bottom')
        plt.savefig(f'{path}/{filename}.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
        if export_eps:
            plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
            os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
            os.system(f"rm {path}/{filename}.pdf")
        plt.close()

def histograms(datasets, path='.', export_eps=False):
    """
    Histograms
    
    Arguments:
    - datasets {dict} -- Dictionary with datasets
    - path {str} -- Path to save the plots
    """
    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    # bold title
    plt.rcParams['axes.titleweight'] = 'bold'
    # remove space between axis label and axis
    plt.rcParams['axes.titlepad'] = 0
    # remove space between tick labels and axis
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['ytick.major.size'] = 0
    # more space between subplots
    plt.rcParams['figure.subplot.hspace'] = 0.5
    plt.rcParams['figure.subplot.wspace'] = 0.15
    # title and axis labels font size
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['axes.labelsize'] = 7
    # tick labels font size
    plt.rcParams['xtick.labelsize'] = 5
    plt.rcParams['ytick.labelsize'] = 5

    for name, dataset in datasets.items():
        filename = f"hist-{name}"
        plt.figure(figsize=(10, 10))
        # add space betwwen subplots
        for j in range(len(dataset.columns)):
            ax = plt.subplot((len(dataset.columns)//4)+1, 4, j+1)
            sns.histplot(dataset.iloc[:, j], ax=ax, bins=50)
            ax.set_title(dataset.columns[j])
            ax.set_xlabel('')
            ax.set_ylabel('')
        plt.savefig(f'{path}/{filename}.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
        if export_eps:
            plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
            os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
            os.system(f"rm {path}/{filename}.pdf")
        plt.close()

def density(datasets, max_y_vals=None, path='.', export_eps=False, **kwargs):
    """
    Density plots
    
    Arguments:
    - datasets {dict} -- Dictionary with datasets
    - path {str} -- Path to save the plots
    """
    text_scale = 1 if not kwargs.get("text_scale") else kwargs.get("text_scale")

    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    
    # bold title
    plt.rcParams['axes.titleweight'] = 'bold'
    # remove space between axis label and axis
    plt.rcParams['axes.titlepad'] = 0
    # remove space between tick labels and axis
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['ytick.major.size'] = 0
    # more space between subplots
    plt.rcParams['figure.subplot.hspace'] = 0.2
    plt.rcParams['figure.subplot.wspace'] = 0.2
    # title and axis labels font size
    plt.rcParams['axes.titlesize'] = 7 * text_scale
    plt.rcParams['axes.labelsize'] = 7 * text_scale
    # tick labels font size
    plt.rcParams['xtick.labelsize'] = 5 * text_scale
    plt.rcParams['ytick.labelsize'] = 5 * text_scale
    # legend font size
    plt.rcParams['legend.fontsize'] = 4 * text_scale
    plt.rcParams['legend.title_fontsize'] = 4 * text_scale

    for name, dataset in datasets.items():
        filename = f"density-{name}"
        plt.figure(figsize=(10, 12)) if len(dataset.columns) > 5 else plt.figure(figsize=(len(dataset.columns)*5, 3))
        # Plot density using gaussian_kde with transformed data
        class_0 = dataset[dataset['fraud_bool'] == 0]
        class_1 = dataset[dataset['fraud_bool'] == 1]
        # drop class label
        class_0 = class_0.drop(columns=['fraud_bool'])
        class_1 = class_1.drop(columns=['fraud_bool'])
        dataset = dataset.drop(columns=['fraud_bool'])
        # normalize data
        for j, col in enumerate(dataset.columns):
            ax = plt.subplot((len(dataset.columns)//5)+1, 5, j+1) if len(dataset.columns) > 5 else plt.subplot(1, len(dataset.columns), j+1)
            sns.kdeplot(class_0[col], ax=ax, fill=True, label='False')
            sns.kdeplot(class_1[col], ax=ax, fill=True, label='True')
            if max_y_vals:
                ax.set_ylim(0, max_y_vals[j])
            ax.set_title(col)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.legend(title='Fraud')
        plt.savefig(f'{path}/{filename}.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
        if export_eps:
            plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
            os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
            os.system(f"rm {path}/{filename}.pdf")
        plt.close()

def normalized_density(datasets, path='.', export_eps=False):
    """
    Density plots
    
    Arguments:
    - datasets {dict} -- Dictionary with datasets
    - path {str} -- Path to save the plots
    """
    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    
    # bold title
    plt.rcParams['axes.titleweight'] = 'bold'
    # remove space between axis label and axis
    plt.rcParams['axes.titlepad'] = 0
    # remove space between tick labels and axis
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['ytick.major.size'] = 0
    # more space between subplots
    plt.rcParams['figure.subplot.hspace'] = 0.5
    plt.rcParams['figure.subplot.wspace'] = 0.15
    # title and axis labels font size
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['axes.labelsize'] = 7
    # tick labels font size
    plt.rcParams['xtick.labelsize'] = 5
    plt.rcParams['ytick.labelsize'] = 5
    # legend font size
    plt.rcParams['legend.fontsize'] = 4
    plt.rcParams['legend.title_fontsize'] = 4

    for name, dataset in datasets.items():
        filename = f"normalized_density-{name}"
        plt.figure(figsize=(10, 12))
        # Plot density using gaussian_kde with transformed data
        class_0 = dataset[dataset['fraud_bool'] == 0]
        class_1 = dataset[dataset['fraud_bool'] == 1]
        # drop class label
        class_0 = class_0.drop(columns=['fraud_bool'])
        class_1 = class_1.drop(columns=['fraud_bool'])
        dataset = dataset.drop(columns=['fraud_bool'])
        # normalize data
        class_0_norm = MinMaxScaler().fit_transform(class_0)
        class_1_norm = MinMaxScaler().fit_transform(class_1)
        for j in range(len(dataset.columns)):
            ax = plt.subplot((len(dataset.columns)//5)+1, 5, j+1)
            sns.kdeplot(class_0_norm[:, j], ax=ax, fill=True, label='False')
            sns.kdeplot(class_1_norm[:, j], ax=ax, fill=True, label='True')
            ax.set_title(dataset.columns[j])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.legend(title='Fraud')
        plt.savefig(f'{path}/{filename}.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
        if export_eps:
            plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
            os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
            os.system(f"rm {path}/{filename}.pdf")
        plt.close()

def pca(datasets, path='.', export_eps=False):
    """
    PCA Projection to 2D
    
    Arguments:
    - datasets {dict} -- Dictionary with datasets
    - path {str} -- Path to save the plots
    """
    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    for name, dataset in datasets.items():
        filename = f"pca-{name}"
        # Remove class label
        y = dataset['fraud_bool']
        X = dataset.drop(columns=['fraud_bool'])
        # Standardize the data
        X = StandardScaler().fit_transform(X)
        # PCA Projection to 2D
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)
        principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        final_df = pd.concat([principal_df, y], axis=1)
        # Plot PCA
        plt.figure(figsize=(10, 10))
        sns.jointplot(data=final_df, x='PC1', y='PC2', hue='fraud_bool')
        plt.savefig(f'{path}/{filename}.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
        if export_eps:
            plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
            os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
            os.system(f"rm {path}/{filename}.pdf")
        plt.close()

def tsne(datasets, path='.', export_eps=False):
    """
    tSNE Projection to 2D

    Arguments:
    - datasets {dict} -- Dictionary with datasets
    - path {str} -- Path to save the plots
    """
    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    for name, dataset in datasets.items():
        filename = f"tsne-{name}"
        # Remove class label
        y = dataset['fraud_bool']
        X = dataset.drop(columns=['fraud_bool'])
        # Standardize the data
        X = StandardScaler().fit_transform(X)
        # tSNE Projection to 2D
        tsne = TSNE(n_components=2)
        principal_components = tsne.fit_transform(X)
        principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        final_df = pd.concat([principal_df, y], axis=1)
        # Plot PCA
        plt.figure(figsize=(10, 10))
        sns.jointplot(data=final_df, x='PC1', y='PC2', hue='fraud_bool')
        plt.savefig(f'{path}/{filename}.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
        if export_eps:
            plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
            os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
            os.system(f"rm {path}/{filename}.pdf")
        plt.close()

def mds(datasets, max_samples=10000, seed=42, path='.', export_eps=False):
    """
    MDS Projection to 2D

    Arguments:
    - datasets {dict} -- Dictionary with datasets
    - max_samples {int} -- Maximum number of samples
    - seed {int} -- Random seed
    - path {str} -- Path to save the plots
    """
    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    for name, dataset in datasets.items():
        filename = f"mds-{name}"
        #sample data
        dataset = sample_data(dataset, max_samples, seed)
        # Remove class label
        y = dataset['fraud_bool']
        X = dataset.drop(columns=['fraud_bool'])
        # Standardize the data
        X = StandardScaler().fit_transform(X)
        # MDS Projection to 2D
        mds = MDS(n_components=2)
        principal_components = mds.fit_transform(X)
        principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        final_df = pd.concat([principal_df, y], axis=1)
        # Plot PCA
        plt.figure(figsize=(10, 10))
        sns.jointplot(data=final_df, x='PC1', y='PC2', hue='fraud_bool')
        plt.savefig(f'{path}/{filename}.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
        if export_eps:
            plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
            os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
            os.system(f"rm {path}/{filename}.pdf")
        plt.close()

def pairplots(datasets, max_samples=10000, seed=42, path='.', export_eps=False):
    """
    Pairplots

    Arguments:
    - datasets {dict} -- Dictionary with datasets
    - max_samples {int} -- Maximum number of samples
    - seed {int} -- Random seed
    - path {str} -- Path to save the plots
    """
    sns.set_theme(style="darkgrid")
    sns.set_context("paper")
    for name, dataset in datasets.items():
        filename = f"pair-{name}"
        dataset = sample_data(dataset, max_samples, seed)
        plt.figure(figsize=(20, 20))
        sns.pairplot(dataset, hue="fraud_bool")
        plt.savefig(f'{path}/{filename}.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
        if export_eps:
            plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
            os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
            os.system(f"rm {path}/{filename}.pdf")
        plt.close()
