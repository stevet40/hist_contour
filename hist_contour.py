import numpy as np
import matplotlib.pyplot as plt

def hist_levels(x, y, w=None, bins=64, levels=[0.95, 0.68]):
    """
    Compute contour levels from a 2D histogram.

    Parameters
    ----------
    x : array-like
        x coordinates of the samples
    y : array-like
        y coordinates of the samples
    w : array-like, optional
        Weights of the samples
    bins : int or (int, int) or array-like or (array, array), optional
        Binning strategy in any format accepted by numpy.histogram2d.
        If int, will use equal number of histogram bins in both directions.
        If (int, int) will use different number in both directions
        If array-like, will use the array as bin edges for both directions.
        If (array, array), will use a different set of edges in both directions.
        Default is 64.
    levels : array-like, optional
        Credible intervals to estimate levels of. Default: [0.95, 0.68]

    Returns
    -------
    x_centres : numpy.array
        Bin centres in x direction
    y_centres : numpy.array
        Bin centres in y direction
    counts : numpy.array
        Bin counts
    heights : list
        Bin counts corresponding to levels
    x_edges : numpy.array
        Bin edges in x direction
    y_edges : numpy.array
        Bin edges in y direction
    """
    # check inputs (not exhaustive)
    if len(x) != len(y):
        raise ValueError("x and y must have same length! Found {:d} and {:d}"
            .format(len(x), len(y)))
    elif w is not None and len(w) != len(x):
        raise ValueError("w and x, y must have same length! Found {:d} and {:d}"
            .format(len(w), len(x)))
    levels = np.array(levels)
    if np.any(levels < 0.0) or np.any(levels > 1.0):
        raise ValueError("levels must be within 0 and 1!")

    # construct histogram
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins, weights=w)
    # find bin centres
    x_centres = 0.5*(x_edges[1:] + x_edges[:-1])
    y_centres = 0.5*(y_edges[1:] + y_edges[:-1])
    # cumulative sum of bin counts
    # go in ascending order of bin count
    counts_sorted = np.sort(counts, axis=None)
    counts_summed = np.cumsum(counts_sorted)
    # cumulative counts normalised by total
    # i.e. counts_summed[i] = fraction of samples
    # accumulated up to the ith largest bin
    counts_summed = counts_summed/counts_summed[-1]
    # find bin counts corresponding to desired credible intervals
    heights = [counts_sorted[np.argmin(np.fabs(counts_summed - (1-l)))]
                 for l in levels]
    # return info needed for plotting
    return x_centres, y_centres, counts.T, heights, x_edges, y_edges

def hist_contour(x, y, w=None, bins=64, levels=[0.95, 0.68], ax=None,
        figsize=(6.4, 6.4), colour="#8ACE00", linewidth=2, linestyle="-",
        xlabel=None, ylabel=None):
    """
    Plot contours based on a 2D histogram.

    Wrapper for hist_levels.

    Parameters
    ----------
    x : array-like
        x coordinates of the samples
    y : array-like
        y coordinates of the samples
    w : array-like, optional
        Weights of the samples
    bins : int or (int, int) or array-like or (array, array), optional
        Binning. See hist_levels.
    levels : array-like, optional
        Credible intervals to estimate levels of. Default: [0.95, 0.68]
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot onto
    figsize : (float, float), optional
        Figure size, if no axis provided. Default: (6.4, 6.4).
    colour : str, optional
        Colour to draw the contours. Default: "#8ACE00"
    linewidth : float, optional
        Contour linewidth. Default: 2
    linestyle : str, optional
        Contour linestyle. Default: "-"
    xlabel : str, optional
        Label for the x axis. Default: None (no label)
    ylabel : str, optional
        Label for the y axis. Default: None (no label)

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis the contours were drawn on.
    """

    # get the contour levels
    x_grid, y_grid, z, l, _, _ = hist_levels(x, y, w, bins, levels)

    # plot
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    ax.contour(x_grid, y_grid, z, levels=l, colors=colour, 
        linewidths=linewidth, linestyles=linestyle)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # return the axis
    return ax
