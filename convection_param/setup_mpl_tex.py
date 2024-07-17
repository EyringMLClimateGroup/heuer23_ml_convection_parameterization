import matplotlib.pyplot as plt

def setup_mpl_tex(font_family='STIXGeneral'):
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2.colors)
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": font_family,
        # "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": 'stix',
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.labelsize": 8,
    }
    plt.rcParams.update(tex_fonts)

def set_size(width, ratio='default', fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'fulla4':
        width_mm = 190
    elif width == 'textwidth':
        # width_pt = 397.4849
        width_mm = 139.68
    elif width == 'halftextwidth':
        width_mm = 139.68 / 2
    elif width == 'halfa4':
        # width_pt = 595 / 2
        width_mm = 95
    else:
        # width_mm = 95 * (1 + width) # scaling btw. halfa4 and a4
        width_mm = width

    # Width of figure (in pts)
    # fig_width_pt = width_pt * fraction
    fig_width_mm = width_mm * fraction
    # Convert from pt to inches
    # inches_per_pt = 1 / 72.27
    inches_per_mm = 1 / 25.4 # inches mm-1

    if ratio == 'golden':
        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        golden_ratio = (5**.5 - 1) / 2
    elif ratio == 'default':
        ratio = 4.8 / 6.4

    # Figure width in inches
    # fig_width_in = fig_width_pt * inches_per_pt
    fig_width_in = fig_width_mm * inches_per_mm
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def get_ax_size(ax, fig):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height