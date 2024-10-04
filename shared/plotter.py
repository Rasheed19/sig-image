import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np

from shared.helper import (
    get_rcparams,
)

# from shared.signature import calc_path_signature

plt.rcParams.update(get_rcparams())


def set_size(
    width: float | str = 360.0,
    fraction: float = 1.0,
    subplots: tuple = (1, 1),
    adjust_height: float | None = None,
) -> tuple:
    # for general use
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    if adjust_height is not None:
        golden_ratio += adjust_height

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def plot_training_pipeline_history(
    history: dict, epoch: int, model_mode: str, batch_size: int
) -> None:
    fig = plt.figure(figsize=set_size(subplots=(1, 2)))
    fig_labels = ("a", "b")
    epoch_array = np.arange(epoch) + 1

    for i, m in enumerate(["loss", "accuracy"]):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.text(
            x=-0.1,
            y=1.2,
            s=r"\bf {}".format(fig_labels[i]),
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )

        for d, c, style in zip(["train", "test"], ["darkcyan", "crimson"], ["-", "--"]):
            ax.plot(
                epoch_array,
                history[f"{d}_{m}"],
                color=c,
                label=f"{d}".capitalize(),
                linestyle=style,
            )
        ax.set_xlabel("Epochs")
        ax.set_ylabel(m.capitalize())
        ax.xaxis.set_major_locator(tck.MaxNLocator(integer=True))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2))

    plt.savefig(
        fname=f"./plots/history_plot_model={model_mode}_batch_size={batch_size}_epochs={epoch}.pdf",
        bbox_inches="tight",
    )

    return None
