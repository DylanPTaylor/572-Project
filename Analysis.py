from itertools import count
from datetime import datetime

from networkx.algorithms.bipartite import cluster
import Functions
import Youtube
import json
import networkx as nx
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

HOME = Functions.Home()

# region Probabilities


def crunch_probabilities(nodes):
    # Expect nodes as a dictionary of dictionaries, top level keys are video ids
    # counter = 0
    # This is also a dictionary of dictionaries, keys are genre names, values are dictionaries of the probability of
    # seeing each genre
    genre_probability_sets = {}
    genre_counters = {}
    for node in nodes:
        this_nodes_probabilities = nodes[node]["probabilities"]
        this_nodes_genre = nodes[node]["genre"]

        # Error checking: skip ones who dont sum to zero
        sum = 0
        for prob in this_nodes_probabilities:
            sum += float(this_nodes_probabilities[prob])

        if sum > 1.1 or sum < 0.9:
            # print(str(counter)+":"+str(sum))
            # counter += 1
            continue

        if this_nodes_genre not in genre_probability_sets:
            genre_probability_sets[this_nodes_genre] = {}
            genre_counters[this_nodes_genre] = 0

        genre_counters[this_nodes_genre] += 1

        for this_genre_id in this_nodes_probabilities:
            if this_genre_id in genre_probability_sets[this_nodes_genre]:
                genre_probability_sets[this_nodes_genre][this_genre_id] += this_nodes_probabilities[this_genre_id]
            else:
                genre_probability_sets[this_nodes_genre][this_genre_id] = this_nodes_probabilities[this_genre_id]

    # Normalize the probabilites we just summed
    for genre in genre_probability_sets:
        probabilities = genre_probability_sets[genre]

        for g in probabilities:
            normalized_probability = probabilities[g]/genre_counters[genre]

            probabilities[g] = normalized_probability

    return genre_probability_sets


def translate(sets):
    for genre in sets:
        probabilities = sets[genre]
        new_probs = {}

        for id in probabilities:
            genre_name = Youtube.translate_genre_id_to_name(id)
            new_probs[genre_name] = probabilities[id]

        sets[genre] = new_probs


def write_probability_file(probs, path):
    column_header = []
    for genre in probs:
        for g in probs[genre]:
            if g not in column_header:
                column_header.append(g)

    if os.path.exists(HOME+path):
        os.remove(HOME+path)

    with open(HOME + path, "w+") as outfile:
        outfile.write("Genre")
        for genre in column_header:
            outfile.write(","+genre)
        outfile.write("\n")
        for genre in probs:
            outfile.write(genre)
            line = [0]*len(column_header)
            for g in probs[genre]:
                line[column_header.index(g)] = probs[genre][g]

            for word in line:
                outfile.write(",{:.8f}".format(word))

            outfile.write("\n")


def probability_analysis(graph):

    probs = crunch_probabilities(dict(graph.nodes()))

    translate(probs)

    write_probability_file(probs, "\\AnalysisData\\genre_probabilities.csv")

# endregion

# region Edge Weights

# endregion

# region Degree


def indegree_by_genre(graph, genres):
    genre_degree_sets = {}
    genre_max_degrees = {}
    nodes = graph.nodes()
    for genre in genres:

        nodes_in_this_genre = [n for n in nodes if nodes[n]["genre"] == genre]
        in_degrees = dict(graph.in_degree(nodes_in_this_genre, "weight"))

        number_of_nodes = len(in_degrees)

        sum = 0
        for node in in_degrees:
            sum += in_degrees[node]

        average = sum / number_of_nodes

        genre_degree_sets[genre] = average
        genre_max_degrees[genre] = max(in_degrees.values())

    return genre_degree_sets, genre_max_degrees


def clustering_by_genre(graph, genres):
    cluster_coefficients = {}
    nodes = graph.nodes()
    for genre in genres:

        nodes_in_this_genre = [n for n in nodes if nodes[n]["genre"] == genre]

        nodes_coefficients = nx.clustering(
            graph, nodes_in_this_genre, "weight")

        sum = 0
        number_of_nodes = len(nodes_coefficients)
        for node in nodes_coefficients:
            sum += nodes_coefficients[node]

        cluster_coefficients[genre] = (sum / number_of_nodes)*100

    return cluster_coefficients


def degree_distribution_by_genre(graph, genres):
    nodes = graph.nodes()
    for genre in genres:
        nodes_in_this_genre = [n for n in nodes if nodes[n]["genre"] == genre]
        degrees = [graph.in_degree(node, "weight")
                   for node in nodes_in_this_genre]
        kmin = min(degrees)
        kmax = max(degrees)
        # Get 100 logarithmically spaced bins between kmin and kmax
        bin_edges = np.logspace(np.log10(kmin+1), np.log10(kmax), num=10)

        # histogram the data into these bins
        density, _ = np.histogram(degrees, bins=bin_edges, density=True)

        fig = plt.figure(figsize=(6, 4))

        # "x" should be midpoint (IN LOG SPACE) of each bin
        log_be = np.log10(bin_edges)
        x = 10**((log_be[1:] + log_be[:-1])/2)

        plt.loglog(x, density, marker='o', linestyle='none')
        plt.xlabel(r"degree $k$", fontsize=16)
        plt.ylabel(r"$P(k)$", fontsize=16)

        # remove right and top boundaries because they're ugly
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # Show the plot
        plt.savefig(
            HOME+"\\AnalysisData\\Degree Distribution\\Deg_Dist_"+genre+".png")


def degree_distribution(graph):
    nodes = graph.nodes()
    degrees = [graph.in_degree(node, "weight")
               for node in nodes]
    kmin = min(degrees)
    kmax = max(degrees)
    # Get 100 logarithmically spaced bins between kmin and kmax
    bin_edges = np.logspace(np.log10(kmin+1), np.log10(kmax), num=15)

    # histogram the data into these bins
    density, _ = np.histogram(degrees, bins=bin_edges, density=True)

    fig = plt.figure(figsize=(6, 4))

    # "x" should be midpoint (IN LOG SPACE) of each bin
    log_be = np.log10(bin_edges)
    x = 10**((log_be[1:] + log_be[:-1])/2)

    plt.loglog(x, density, marker='o', linestyle='none')
    plt.xlabel(r"degree $k$", fontsize=16)
    plt.ylabel(r"$P(k)$", fontsize=16)

    # remove right and top boundaries because they're ugly
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Show the plot
    plt.savefig(
        HOME+"\\AnalysisData\\Degree Distribution\\Deg_Dist_ALL.png")


def write_degree_file(genres, avgs, maxs, coeffs, path):

    if os.path.exists(HOME+path):
        os.remove(HOME+path)

    with open(HOME + path, "w+") as outfile:
        outfile.write(
            "Genre,Average In Degree,Highest In Degree,Clustering Coefficient\n")
        for genre in genres:
            outfile.write(genre+","+str(avgs[genre]) +
                          ","+str(maxs[genre])+","+str(coeffs[genre])+"\n")


def degree_analysis(graph, genres):

    avg_indegrees, max_degrees = indegree_by_genre(graph, genres)

    cluster_coefficients = clustering_by_genre(graph, genres)

    write_degree_file(genres, avg_indegrees, max_degrees,
                      cluster_coefficients, "\\AnalysisData\\degree_analysis.csv")

    #degree_distribution_by_genre(graph, genres)
    degree_distribution(graph)


# endregion

# region community clumping?
# endregion
if __name__ == "__main__":
    relative_path = "\\Level2\\07-11-2021\\"

    graph = Functions.load_graph(HOME+relative_path)
    genres = Functions.load_genres_from_file()
    probability_analysis(graph)
    degree_analysis(graph, genres)

"""
TODO
Add analysis that looks at the probability of getting from genre A to genre B in k hops.
Set it up as a graph, probabilities are edge weights, run path finding algorithms
"""
