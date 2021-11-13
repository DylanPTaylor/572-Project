from itertools import count
from datetime import datetime

from networkx.algorithms.bipartite import cluster
from networkx.algorithms.shortest_paths.generic import shortest_path
import Functions
import Youtube
import json
import networkx as nx
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter
from operator import indexOf, itemgetter

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


def write_genre_by_genre(probs, path):
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

    return probs

# endregion

# region Degree
def count_nodes(graph,genres):
    nodes = graph.nodes()
    genres_counters = {}
    for genre in genres:
        
        nodes_in_this_genre = [n for n in nodes if nodes[n]["genre"] == genre]
        number_of_nodes = len(nodes_in_this_genre)
        genres_counters[genre] = number_of_nodes
    return genres_counters

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


"""
The following code we taken from

https://stackoverflow.com/questions/53958700/plotting-the-degree-distribution-of-a-graph-using-nx-degree-histogram

"""


def degree_distribution_by_genre(graph, genres):
    nodes = graph.nodes()
    for genre in genres:
        nodes_in_this_genre = [n for n in nodes if nodes[n]["genre"] == genre]
        degseq = [graph.in_degree(node, "weight")
                  for node in nodes_in_this_genre]
        dmax = max(degseq)+1
        freq = [0 for d in range(dmax)]
        for d in degseq:
            freq[d] += 1

        degrees = range(len(freq))
        plt.figure(figsize=(12, 8))
        plt.loglog(range(len(freq)), freq, 'go-', label='in-degree')

        plt.xlabel('Degree')
        plt.ylabel('Frequency')

        # Show the plot
        plt.savefig(
            HOME+"\\AnalysisData\\Degree Distribution\\"+genre+".png")


"""
The following code we taken from

https://stackoverflow.com/questions/53958700/plotting-the-degree-distribution-of-a-graph-using-nx-degree-histogram

as it was in the prvious method
"""


def degree_dist_all(G):
    nodes = G.nodes()
    in_degree = dict(G.in_degree())
    degseq = [in_degree.get(k, 0) for k in nodes]
    dmax = max(degseq)+1
    freq = [0 for d in range(dmax)]
    for d in degseq:
        freq[d] += 1

    degrees = range(len(freq))
    plt.figure(figsize=(12, 8))
    plt.loglog(range(len(freq)), freq, 'go-', label='in-degree')

    plt.xlabel('Degree')
    plt.ylabel('Frequency')

    plt.savefig(
        HOME+"\\AnalysisData\\Degree Distribution\\ALL.png")
# not quite right


def degree_analysis(graph, genres):

    avg_indegrees, max_degrees = indegree_by_genre(graph, genres)

    cluster_coefficients = clustering_by_genre(graph, genres)

    degree_dist_all(graph)
    degree_distribution_by_genre(graph, genres)

    return avg_indegrees, max_degrees, cluster_coefficients


# endregion

# region Edge Weights
def fewest_hops_to_exit(graph, genres):
    all_nodes = graph.nodes()
    average_hops_by_genre = {}
    for genre in genres:
        genre_nodes = [n for n in all_nodes if all_nodes[n]["genre"] == genre]

        sum = 0
        count = len(genre_nodes)

        for node in genre_nodes:
            all_paths = nx.shortest_path(
                graph, node, None, "weight", "dijkstra")

            for n in genre_nodes:
                try:
                    del all_paths[n]
                except:
                    continue

            all_paths = all_paths.values()

            try:
                sum += min([len(i) for i in all_paths])
            except:
                count -= 1

        average = sum/count

        average_hops_by_genre[genre] = average
    return average_hops_by_genre


def fewest_hops_to_enter(graph, genres):
    all_nodes = graph.nodes()
    average_hops_by_genre = {}
    all_paths = nx.shortest_path(graph, weight='weight')
    
    for genre in genres:
        genre_nodes = [n for n in all_nodes if all_nodes[n]["genre"] == genre]

        sum = 0
        count = len(all_nodes) - len(genre_nodes)

        for starting_node in all_paths:
            if all_nodes[starting_node]['genre'] == genre:
                continue
            
            #these are already sorted by list length
            this_nodes_paths = all_paths[starting_node]
            for target_node in this_nodes_paths:
                if all_nodes[target_node]['genre'] == genre:
                    sum += len(all_paths[starting_node][target_node])
                    break
                elif target_node == list(this_nodes_paths)[-1]:
                    count -= 1


        average = sum/count

        average_hops_by_genre[genre] = average
    return average_hops_by_genre
# endregion

# region Community detection


def from_one_to_the_other(graph, genres):
    nodes = graph.nodes()
    average_distance = {}
    paths = nx.shortest_path(graph, weight="weight")
    for genre_A in genres:
        genre_A_nodes = [n for n in nodes if nodes[n]["genre"] == genre_A]

        for genre_B in genres:
            if genre_A == genre_B:
                continue

            genre_B_nodes = [n for n in nodes if nodes[n]["genre"] == genre_B]

            total_sum = 0
            no_path_count = 0
            # sum the path lengths from any node in genre A to any node in genre B
            for nodeA in genre_A_nodes:
                node_A_sum = 0
                count = len(genre_B_nodes)
                for nodeB in genre_B_nodes:
                    try:
                        node_A_sum += len(paths[nodeA][nodeB])
                    except:
                        count -= 1
                try:
                    node_A_average = node_A_sum/count
                    total_sum += node_A_average
                except:
                    no_path_count += 1
                    continue

            average = total_sum/(len(genre_A_nodes)-no_path_count)
            try:
                average_distance[genre_A][genre_B] = average
            except:
                average_distance[genre_A] = {}
                average_distance[genre_A][genre_B] = average

    return average_distance

# def girvan_newman():
# endregion

# region Centrality


def betweenness(graph, genres):
    genre_centrality = {}
    counters = {}
    all_nodes = graph.nodes()
    centrality = nx.betweenness_centrality(
        G=graph, k=None, normalized=True, weight='weight')
    for node in centrality:
        node_genre = all_nodes[node]['genre']
        try:
            genre_centrality[node_genre] += centrality[node]
            counters[node_genre] += 1
        except:
            genre_centrality[node_genre] = centrality[node]
            counters[node_genre] = 1

    for genre in genre_centrality:
        genre_centrality[genre] = (
            genre_centrality[genre] / counters[genre]) * 1000

    return genre_centrality
# endregion


def write_by_genre(genres, path, avg_in_deg, maxs, coeffs, hops):

    if os.path.exists(HOME+path):
        os.remove(HOME+path)

    with open(HOME + path, "w+") as outfile:
        outfile.write(
            "Genre,Average In Degree,Highest In Degree,Clustering Coefficient,Average Hops To Exit\n")
        for genre in genres:
            outfile.write(genre+","+str(avg_in_deg[genre]) + ","+str(
                maxs[genre])+","+str(coeffs[genre])+","+str(hops[genre])+"\n")


def write_by_genre(genres, path, dict, name):

    if os.path.exists(HOME+path):
        os.remove(HOME+path)

    with open(HOME + path, "w+") as outfile:
        outfile.write(
            "Genre,"+name+"\n")
        for genre in genres:
            outfile.write(genre+","+str(dict[genre]) + "\n")


if __name__ == "__main__":
    relative_path = "\\Level2\\09-11-2021\\"

    graph = Functions.load_graph(HOME+relative_path)
    inverted_graph = Functions.load_graph(HOME+relative_path, True)
    genres = Functions.load_genres_from_file()

    enter_hops = fewest_hops_to_enter(inverted_graph, genres)
    
    write_by_genre(genres, "\\AnalysisData\\hops_to_enter.csv",
                   enter_hops, "Average Fewest Hops To Enter")
    
    """
    
    exit_hops = fewest_hops_to_exit(inverted_graph, genres)
    avg_indegrees, max_degrees, cluster_coefficients = degree_analysis(
        graph, genres)

    avg_dist_btwn_genres = from_one_to_the_other(
        inverted_graph, genres)

    between = betweenness(inverted_graph, genres)

    probs = probability_analysis(graph)
    
    num_of_nodes = count_nodes(graph, genres)

    write_genre_by_genre(probs, "\\AnalysisData\\genre_probabilities.csv")
    write_genre_by_genre(avg_dist_btwn_genres,
                         "\\AnalysisData\\distance.csv")
    write_by_genre(genres, "\\AnalysisData\\node_count.csv",num_of_nodes, "Number of Nodes")
    write_by_genre(genres, "\\AnalysisData\\betweeness.csv",
                   between, "Average Betweeness")
    write_by_genre(genres, "\\AnalysisData\\in_degrees.csv",
                   avg_indegrees, "Average In-Degree")
    write_by_genre(genres, "\\AnalysisData\\max_degrees.csv",
                   max_degrees, "Highest In-Degree")
    write_by_genre(genres, "\\AnalysisData\\clusternig_coefficient.csv",
                   cluster_coefficients, "Average Clustering Coefficient")
    write_by_genre(genres, "\\AnalysisData\\hops_to_exit.csv",
                   exit_hops, "Average Fewest Hops To Exit")
    """

"""
TODO

Add analysis that looks at the probability of getting from genre A to genre B in k hops.
Set it up as a graph, probabilities are edge weights, run path finding algorithms

shortest path from any node in genre A to any node in genre B, based on edge weights 
    -> shorter paths use higher edge weights, as nodes were more likely to click on those videos

simrank_similarity: compute average similarity of nodes in a genre

"""
