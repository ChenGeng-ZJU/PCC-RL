from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for x in ax.spines:
        ax.spines[x].set_visible(False)
#     ax.spines.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

tsss = np.array([ 0.1,  0.2,  0.4,  0.8,  1. ,  1.5,  2. ,  4. ,  8. , 10. , 12. , 15. , 17. , 20. , 22.])
bws = np.array([6.0e-01, 8.0e-01, 1.0e+00, 1.5e+00, 2.0e+00, 2.5e+00, 3.0e+00, 4.0e+00, 5.0e+00, 7.0e+00, 1.0e+01, 1.5e+01, 2.0e+01, 2.5e+01, 4.0e+01, 6.0e+01, 8.0e+01, 1.0e+02])

def draw_bo(csv_file, bo):
    data = pd.read_csv(csv_file)
    mat = np.zeros((bws.shape[0], tsss.shape[0]))
    for bwi, bw in enumerate(bws):
        for tsi, tss in enumerate(tsss):
            dt = data[(data['bw'] == bw) & (data['ts'] == tss)]['deltar']
    #         print(len(dt))
            try:
                mat[bwi, tsi] = dt.iloc[0]
            except:
                mat[bwi, tsi] = 10
    # plt.figure(figsize=(10, 10))
    mat2 = mat.copy()
    mat2[mat2 > 10] = 10
    fig, ax = plt.subplots(figsize=(10, 10))
    im, cbar = heatmap(mat2, bws, tss, ax, cbarlabel="Aurora's Reward - BBR's Reward")
    texts = annotate_heatmap(im, valfmt="{x:.1f}")
    ax.set_title("The difference between aurora's reward and bbr's reward, with bo {}".format(bo))
    fig.tight_layout()
    ax.set_xlabel("T_s")
    ax.set_ylabel("Bandwidth")
    # plt.show()
    fig.savefig('heatmap_with_bo{}.png'.format(bo))

if __name__ == "__main__":
    import argparse
    parser = ArgumentParser()
    parser.add_argument('--file', type=str)
    parser.add_argument('--bo', type=int)
    args = parser.parse_args()
    draw_bo(args.file, args.bo)