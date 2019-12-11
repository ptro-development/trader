from pylab import subplot, plot, grid, title, ion, clf
from pylab import show, setp, rcParams, imshow, draw


def plot_data(x, y, subplot_id):
    subplot(subplot_id)
    plot(x, y, "b--")
    grid(True)
    # title("Progress of network error.")
    return subplot_id + 1


def plot_distribution(data, subplot_id):
    sp = subplot(subplot_id)
    grid(True)
    number_of_bars = 60
    n, bins, patches = sp.hist(
        data, number_of_bars, normed=1, histtype="bar"
    )
    setp(patches, "facecolor", "g", "alpha", 0.75)
    return subplot_id + 1


def plot_two_vectors(first_and_second, subplot_id):
    for i, (first, second) in enumerate(first_and_second):
        subplot(subplot_id)
        plot(first, "b--", second, "k-")
        grid(True)
        subplot_id += 1
    return subplot_id


def plot_plus_correlations_for_sample(data, sample, subplot_num, limit=1):
    two_vectors = []
    for index, position in enumerate(sample["+correlation_positions"]):
        if index < limit:
            two_vectors.append(
                [
                    sample["sample_data"],
                    data[sample["+correlation_positions"][index]: sample["+correlation_positions"][index] + len(sample["sample_data"])]
                ]
            )
    plot_two_vectors(two_vectors, subplot_num)
