import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from snntorch import spikeplot as splt

def plot_spk_mem_spk(spk_in, mem, spk_out, filename="example_lif-spk_mem_spk.png", path=".", export_eps=False):
    """
    Plot the input spikes, membrane potential and output spikes of a LIF neuron

    Parameters
    ----------
        spk_in : torch.Tensor
            input spikes
        mem : torch.Tensor
            membrane potential
        spk_out : torch.Tensor
            output spikes
        path : str
            path to save the plot
        filename : str
            name of the file
        extension : str, optional
            extension of the file (default: "png")
        export_eps: bool, optional
            boolean to export eps file
    """
    extension = filename.split('.')[1]
    filename = filename.split('.')[0]

    sns.set_theme(style="darkgrid")
    sns.set_context("paper")

    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8,6), sharex=True, gridspec_kw = {'height_ratios': [0.4, 1, 0.4]})

    # Plot input current
    splt.raster(spk_in, ax[0], s=400, c="black", marker="|")
    ax[0].set_ylabel("Input Spikes")
    plt.yticks([]) 

    # Plot membrane potential
    ax[1].plot(mem)
    ax[1].set_ylim([0, 1])
    ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
    ax[1].axhline(y=0.5, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk_out, ax[2], s=400, c="black", marker="|")
    plt.ylabel("Output spikes")
    plt.yticks([]) 
    plt.savefig(f'{path}/{filename}.{extension}', dpi=300, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")


def plot_cur_mem_spk(cur, mem, spk, steps, thr_line=False, vline=False, ylim_max2=1.25, filename="example_lif-cur_mem_spk.png", path=".", export_eps=False):
    """
    Plot the input current, membrane potential and output spikes of a LIF neuron

    Parameters
    ----------
        cur : torch.Tensor
            input current
        mem : torch.Tensor
            membrane potential
        spk : torch.Tensor
            output spikes
        path : str
            path to save the plot
        filename : str
            name of the file
        extension : str, optional
            extension of the file (default: "png")
        export_eps: bool, optional
            boolean to export eps file
    """
    extension = filename.split('.')[1]
    filename = filename.split('.')[0]

    sns.set_theme(style="darkgrid")
    sns.set_context("paper")

    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8,6), sharex=True, gridspec_kw = {'height_ratios': [1, 1, 0.4]})

    # Plot input current
    ax[0].plot(cur, c="tab:orange")
    ax[0].set_ylim([-0.01, 0.4])
    ax[0].set_xlim([0, steps])
    ax[0].set_ylabel("Input Current ($I$)")

    # Plot membrane potential
    ax[1].plot(mem)
    ax[1].set_ylim([-0.01, ylim_max2]) 
    ax[1].set_ylabel("Membrane Potential ($U$)")
    if thr_line:
        ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time Step")

    # Plot output spike using spikeplot
    splt.raster(spk, ax[2], s=400, c="black", marker="|")
    if vline:
        ax[2].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
    ax[2].set_ylabel("Spikes")
    plt.yticks([]) 
    plt.savefig(f'{path}/{filename}.{extension}', dpi=300, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")


def plot_spk_cur_mem_spk(spk_in, cur, mem, spk, steps, thr_line=False, vline=False, filename="example_lif-spk_cur_mem_spk.png", path=".", export_eps=False, export_pdf=False, **kwargs):
    """
    Plot the input current, membrane potential and output spikes of a LIF neuron

    Parameters
    ----------
        spk_in : torch.Tensor
            input spikes
        cur : torch.Tensor
            input current
        mem : torch.Tensor
            membrane potential
        spk : torch.Tensor
            output spikes
        path : str
            path to save the plot
        filename : str
            name of the file
        extension : str, optional
            extension of the file (default: "png")
        export_eps: bool, optional
            boolean to export eps file
    """
    extension = filename.split('.')[1]
    filename = filename.split('.')[0]
    dpi = kwargs.get("dpi", 1000)
    palette = kwargs.get("palette", "darkgrid")
    sns.set_theme(style=palette)
    sns.set_context("paper")

    # Generate Plots
    fig, ax = plt.subplots(4, figsize=(10,5), sharex=True, gridspec_kw = {'height_ratios': [0.4, 1, 1, 0.4]})

    # Plot input current
    splt.raster(spk_in, ax[0], s=400, c="black", marker="|")
    ax[0].set_ylabel("Input\nSpikes\n")
    ax[0].set_ylim([-0.1, 0.1])
    ax[0].set_yticks([])

    # Plot input current
    ax[1].plot(cur, c="tab:orange")
    ax[1].set_xlim([0, steps])
    ax[1].set_ylabel("Input Current ($I$)")

    # Plot membrane potential
    ax[2].plot(mem)
    ax[2].set_ylabel("Membrane Potential ($U$)")
    if thr_line:
        ax[2].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    plt.xlabel("Time Step")

    # Plot output spike using spikeplot
    splt.raster(spk, ax[3], s=400, c="black", marker="|")
    if vline:
        ax[3].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
    ax[3].set_ylabel("Output\nSpikes\n")
    ax[3].set_ylim([-0.1, 0.1])
    ax[3].set_yticks([])
    if not export_pdf:
        plt.savefig(f'{path}/{filename}.{extension}', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    if export_eps:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        os.system(f"gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile={path}/{filename}.eps {path}/{filename}.pdf")
        os.system(f"rm {path}/{filename}.pdf")
    if export_pdf:
        plt.savefig(f"{path}/{filename}.pdf", format="pdf", dpi=dpi, bbox_inches='tight', pad_inches=0.1)