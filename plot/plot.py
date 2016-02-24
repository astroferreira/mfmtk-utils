

def histograms(param, x, axes, color='b', bins=25, normed=1, alpha=0.5):
    for column, ax, xi in zip(param.T, axes.flat, x):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.hist(column, color=color,
         bins=bins, normed=normed, alpha=alpha)
