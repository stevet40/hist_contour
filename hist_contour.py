import numpy as np
import matplotlib.pyplot as plt

def hist_levels(x, y, w=None, x_bins=32, y_bins=32, levels=[0.95, 0.68],
        x_lims=[None, None], y_lims=[None, None], 
        x_width=None, y_width=None, percentile_lims=False):
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
    x_bins : int or array-like, optional
        Binning strategy for the x direction. Formats accepted:
        int        --> will use a specified number of bins
        array-like --> will use the specified bin edges
        Overridden by x_width.
        Default is 64.
    y_bins : int or array-like, optional
        Binning strategy in the y direction. Overridden by y_width.
    levels : array-like, optional
        Credible intervals to estimate levels of. Default: [0.95, 0.68]
    x_lims : (float, float), optional
        If specified, will only consider samples with x coordinates
        within these limits (so a fixed bin number will place that
        many bins within the range defined by `xlims`, etc.).
        Default is (None, None) --> no limits.
    y_lims : (float, float), optional
        Limits in the y direction.
    x_width : float, optional
        If passed, will override other arguments if needed and will
        force a set bin width in the x direction. Default is None.
    y_width : float, optional
        Forces a set bin width in the y direction.
    percentile_lims : bool, optional
        If True, will interpret `xlims` and `ylims` as fractional
        upper and lower limits; i.e. (0.01, 0.99) would place the
        bounds at the 1st and 99th percentile of the samples.

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
    # lengths should all be the same
    if len(x) != len(y):
        raise ValueError("x and y must have same length!\n" +
            "Found: {:d} and {:d}".format(len(x), len(y)))
    elif w is not None and len(w) != len(x):
        raise ValueError("w and x, y must have same length!\n" 
            "Found: {:d} and {:d}".format(len(w), len(x)))
    # check levels are in [0,1]
    levels = np.array(levels)
    if np.any(levels < 0.0) or np.any(levels > 1.0):
        raise ValueError("levels must be within 0 and 1!\nFound: " + 
            (len(levels)*"{:.2f}").format(*levels))
    # check that only two x and y limits are given
    if len(x_lims) != 2 or len(y_lims) != 2:
        raise ValueError("xlims and ylims must have two elements!\n" +
            "Found: len(xlims)={:d} and len(ylims)={:d}".format(
                len(x_lims), len(y_lims)))
    # sort out the limits
    if x_lims[0] is None:    
        xmin = np.min(x)
    elif percentile_lims is False:
        xmin = x_lims[0]
    else:
        xmin = np.quantile(x, x_lims[0])
    if x_lims[1] is None:
        xmax = np.max(x)
    elif percentile_lims is False:
        xmax = x_lims[1]
    else:
        xmax = np.quantile(x, x_lims[1])
    if y_lims[0] is None:
        ymin = np.min(y)
    elif percentile_lims is False:
        ymin = y_lims[0]
    else:
        ymin = np.quantile(y, y_lims[0])
    if y_lims[1] is None:
        ymax = np.max(y)
    elif percentile_lims is False:
        ymax = y_lims[1]
    else:
        ymax = np.quantile(y, y_lims[1])
    # sort out the bin widths
    if x_width is not None:
        x_range = xmax - xmin
        nx, rem = np.divmod(x_range, x_width)
        # deal with non-integer number of bins in range
        if rem > 0:
            nx += 1
            overflow = 0.5*(x_width - rem)
            xmin -= overflow
            xmax += overflow
        x_bins = np.linspace(xmin, xmax, int(nx)+1)
    if y_width is not None:
        y_range = ymax - ymin
        ny, rem = np.divmod(y_range, y_width)
        if rem > 0:
            ny += 1
            overflow = 0.5*(y_width - rem)
            ymin -= overflow
            ymax += overflow
        y_bins = np.linspace(ymin, ymax, int(ny)+1)

    # mask for points outside the limits
    x_mask = (x < xmin) + (x > xmax)
    y_mask = (y < ymin) + (y > ymax)
    mask = x_mask + y_mask

    # construct histogram
    counts, x_edges, y_edges = np.histogram2d(x[~mask], y[~mask],
        bins=(x_bins, y_bins), weights=w, 
        range=((xmin, xmax), (ymin, ymax)))
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

def hist_contour(x, y, w=None, x_bins=32, y_bins=32, levels=[0.95, 0.68], 
        x_lims=[None, None], y_lims=[None, None], 
        x_width=None, y_width=None, percentile_lims=False,
        ax=None, figsize=(6.4, 6.4), colour="#8ACE00", 
        linewidth=2, linestyle="-", xlabel=None, ylabel=None):
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
    x_bins : int or array-like, optional
        Binning in x direction. See `hist_levels`.
    y_bins : int or array-like, optional
        Binning in y direction. See `hist_levels`.
    levels : array-like, optional
        Credible intervals to estimate levels of. Default: [0.95, 0.68]
    x_lims : (float, float), optional
        Limits in the x direction. See `hist_levels`.
    y_lims : (float, float), optional
        Limits in the y direction. See `hist_levels`.
    x_width : float, optional
        Specified bin width in x direction. See `hist_levels`.
    y_width : float, optional
        Specified bin width in y direction. See `hist_levels`.
    percentile_lims : bool, optional
        If True, will interpret `xlims` and `ylims` as fractional
        upper and lower limits. See `hist_levels`.
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
    x_grid, y_grid, z, l, x_edges, y_edges = hist_levels(x, y, w,
        x_bins=x_bins, y_bins=y_bins, levels=levels, x_lims=x_lims, 
        y_lims=y_lims, x_width=x_width, y_width=y_width,
        percentile_lims=percentile_lims)

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
    ax.set_xlim(np.min(x_edges), np.max(x_edges))
    ax.set_ylim(np.min(y_edges), np.max(y_edges))
    # return the axis
    return ax
