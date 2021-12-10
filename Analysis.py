"""
Author: Dylan Taylor
UCID: 30078900
Date: 08/12/2021

This is the code that processes all the gathered data for our CPSC 572 project at the University of Calgary.

Note:
For Pathology metrics, we do not include nodes without paths when computing averages,
as the lack of a path is more reflective of the lack of data we gathered as opposed to YouTubes
ability to suggest videos.
"""


import Functions
import Youtube
import networkx as nx
import numpy as np
from networkx.algorithms.community import modularity_max as mod_max
import os
import matplotlib.pyplot as plt

HOME = Functions.Home()

# region Probabilities

"""
crunch_probabities takes in a list of nodes which contain 2 attributes:
    probabilities 
    genre

for each node, 
    it checks that the probabilities sum to 1, allowing for small error.
    then add these values to the appropriate genres distribution, 
    ie. for every other genre, sums[genre] += this_nodes_distribution
    
then, for each genre
    divide the sum by the number of nodes 
    so sum(probabilities[genre]) = 1

This returns a cartesian product which is genre x genre
or rather probabilites[A][B] = prob of seeing B when in A.
"""


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


"""
Converts numeric genre to its string equivalent, define by youtube, using data gathered from the API.
"""


def translate(sets):
    for genre in sets:
        probabilities = sets[genre]
        new_probs = {}

        for id in probabilities:
            genre_name = Youtube.translate_genre_id_to_name(id)
            new_probs[genre_name] = probabilities[id]

        sets[genre] = new_probs


"""
Driver method
"""


def probability_analysis(graph):

    probs = crunch_probabilities(dict(graph.nodes()))

    translate(probs)

    return probs

# endregion

# region Degree


"""
Count nodes by genre
"""


def count_nodes(graph, genres):
    nodes = graph.nodes()
    genres_counters = {}
    for genre in genres:

        nodes_in_this_genre = [n for n in nodes if nodes[n]["genre"] == genre]
        number_of_nodes = len(nodes_in_this_genre)
        genres_counters[genre] = number_of_nodes
    return genres_counters


"""
Gets average indegree for each genre
"""


def avg_indegree_by_genre(graph, genres):
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


"""
Get average clustering coefficient for each genre
"""


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

        cluster_coefficients[genre] = (sum / number_of_nodes)

    return cluster_coefficients


"""
The following code we taken from

https://stackoverflow.com/questions/53958700/plotting-the-degree-distribution-of-a-graph-using-nx-degree-histogram


and while it is not the code used for th final graphs, inspiration was taken from it
and so it is left here.
"""


def degree_histogram_directed(G, in_degree=False, out_degree=False):
    """Return a list of the frequency of each degree value.

    Parameters
    ----------
    G : Networkx graph
       A graph
    in_degree : bool
    out_degree : bool

    Returns
    -------
    hist : list
       A list of frequencies of degrees.
       The degree values are the index in the list.

    Notes
    -----
    Note: the bins are width one, hence len(list) can be large
    (Order(number_of_edges))
    """
    nodes = G.nodes()
    if in_degree:
        in_degree = dict(G.in_degree())
        degseq = [in_degree.get(k, 0) for k in nodes]
    elif out_degree:
        out_degree = dict(G.out_degree())
        degseq = [out_degree.get(k, 0) for k in nodes]
    else:
        degseq = [v for k, v in G.degree()]
    dmax = max(degseq)+1
    freq = [0 for d in range(dmax)]
    for d in degseq:
        freq[d] += 1
    return freq


def plot_graph(G, path):

    in_degree_freq = degree_histogram_directed(G, in_degree=True)
    out_degree_freq = degree_histogram_directed(G, out_degree=True)

    plt.figure(figsize=(12, 8))
    plt.loglog(range(len(in_degree_freq)),
               in_degree_freq, 'go-', label='in-degree')
    plt.loglog(range(len(out_degree_freq)),
               out_degree_freq, 'bo-', label='out-degree')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.savefig(path)
    plt.close()


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
    plt.loglog(degrees, freq, label='in-degree')

    plt.xlabel('Degree')
    plt.ylabel('Frequency')

    plt.savefig(
        HOME+"\\AnalysisData\\Degree Distribution\\ALL2.png")


"""
END OF CODE FROM STACK OVERFLOW
"""
"""
Driver method
"""


def degree_analysis(graph, genres):

    avg_indegrees, max_degrees = avg_indegree_by_genre(graph, genres)

    cluster_coefficients = clustering_by_genre(graph, genres)

    degree_dist_all(graph)

    return avg_indegrees, max_degrees, cluster_coefficients


# endregion

# region Edge Weights

"""
Takes in a weighted graph where nodes have a 'genre' attribute, and a list of youtube genres as their string encoding

Returns the average shortest length of paths from all nodes in A to any node in B for every genre A to every genre B!=A
"""


def fewest_hops_to_exit(graph, genres):
    all_nodes = graph.nodes()
    average_hops_by_genre = {}
    for genre in genres:
        # Get nodes in this genre
        genre_nodes = [n for n in all_nodes if all_nodes[n]["genre"] == genre]

        # for computing mean
        sum = 0
        count = len(genre_nodes)

        for node in genre_nodes:
            # get all paths from this node to every other node
            all_paths = nx.shortest_path(
                graph, node, None, "weight", "dijkstra")

            # delete the paths that have targets in the sources genre
            for n in genre_nodes:
                try:
                    del all_paths[n]
                except:
                    continue

            # all_paths[node] = list of paths, so all_paths.values = list of all paths for nodes
            all_paths = all_paths.values()

            # Take the shortest one. We know this is in another genre as the ones ending in
            # the same genre were deleted
            try:
                sum += min([len(i) for i in all_paths])
            except:
                # Must not be a path for this node to exit -> happens in DiGraphs
                # decrease count so zeros dont skew data.
                count -= 1

        average = sum/count
        average_hops_by_genre[genre] = average

    return average_hops_by_genre


"""
Takes in a weighted graph where nodes have a 'genre' attribute, and a list of youtube genres as their string encoding

Returns average shortest length for paths from every node outside genre A, to any node inside a genre
"""


def fewest_hops_to_enter(graph, genres):
    all_nodes = graph.nodes()
    average_hops_by_genre = {}

    # calculate all paths and then index by iterating over nodes
    all_paths = nx.shortest_path(graph, weight='weight')

    for genre in genres:
        # Get nodes in this genre
        genre_nodes = [n for n in all_nodes if all_nodes[n]["genre"] == genre]

        sum = 0
        # Count how many nodes NOT in this genre
        count = len(all_nodes) - len(genre_nodes)

        for starting_node in all_paths:
            # Do not consider nodes already in this genre
            if all_nodes[starting_node]['genre'] == genre:
                continue

            # this_nodes_paths are already sorted by list length
            this_nodes_paths = all_paths[starting_node]

            for target_node in this_nodes_paths:
                # if we come across a target in the genre
                if all_nodes[target_node]['genre'] == genre:
                    # take the path length -> since this_nodes_paths is sorted, the first one is the shortest path
                    sum += len(all_paths[starting_node][target_node])
                    break  # so we only take the shortest path for starting_node

                # Otherwise if this is the last path for starting_node, and target is still not in the genre
                # there is no path from this node -> happens in DiGraph
                elif target_node == list(this_nodes_paths)[-1]:
                    count -= 1  # decrement to not skew results

        average = sum/count

        average_hops_by_genre[genre] = average
    return average_hops_by_genre
# endregion

# region Community detection


"""
Takes in a graph and iterates over its edges. It computes the index for A in relation to B!=A by:
1) Computing the number of ties between nodes in A and other nodes in A
2) Computing the number of ties between nodes in A and nodes in B
3) Finding their difference and dividing by their sum

Computes the EI index for every genre A in relation to every genre B
"""


def ei_index(graph, genres):
    nodes = graph.nodes()
    ei_index = {}
    for genre_A in genres:
        genre_A_nodes = [n for n in nodes if nodes[n]["genre"] == genre_A]

        for genre_B in genres:
            # Only consider other genres
            if genre_A == genre_B:
                continue

            genre_B_nodes = [n for n in nodes if nodes[n]["genre"] == genre_B]

            # Count the ties member of A have with other members of A
            ties_within_genre_A = 0
            # Count the number of ties from node A to genre B
            ties_with_B = 0
            for (a, b) in graph.edges():
                # if an edge connects two nodes in A
                if (a in genre_A_nodes and b in genre_A_nodes):
                    ties_within_genre_A += 1
                # if a edge connects a node in A and a node in B, or vice versa
                elif (a in genre_A_nodes and b in genre_B_nodes) or (a in genre_B_nodes and b in genre_A_nodes):
                    ties_with_B += 1

            total_ties = ties_within_genre_A + ties_with_B

            # Negative values mean A associates with itself more than it associates with
            # B. Most, if not all, values withh be negative since the algorithm heavily
            # favours a videos own genre over any other.
            index = (ties_with_B - ties_within_genre_A) / total_ties
            try:
                ei_index[genre_A][genre_B] = index
            except:
                ei_index[genre_A] = {}
                ei_index[genre_A][genre_B] = index

    return ei_index


"""
Ended up not using this metric, and so it is not commented
but it calculated the average distance from any node in A to any node in B.
"""


def from_one_to_the_other(graph, genres):
    nodes = graph.nodes()
    average_distance = {}
    paths = nx.shortest_path(graph)
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
            try:
                average = total_sum/(len(genre_A_nodes)-no_path_count)
            except:
                average = 0
            try:
                average_distance[genre_A][genre_B] = average
            except:
                average_distance[genre_A] = {}
                average_distance[genre_A][genre_B] = average

    return average_distance


"""
Also did use this one as I simply used the one in gephi
"""


def greedy_modularity(undirected_graph, original_graph):
    set_of_communities = mod_max.greedy_modularity_communities(
        undirected_graph, 'weight')

    genre_scores = {}
    nodes = graph.nodes()
    for community in set_of_communities:
        genre_counters = {}
        for node in community:
            node_genre = nodes[node]['genre']
            try:
                genre_counters[node_genre] += 1
            except:
                genre_counters[node_genre] = 1

        dominating_genre_index = list(genre_counters.values()).index(
            max(genre_counters.values()))
        dominating_genre = list(genre_counters)[dominating_genre_index]
        try:
            genre_scores[dominating_genre] += 1
        except:
            genre_scores[dominating_genre] = 1

        subG = nx.subgraph(original_graph, community)

        Functions.write_graph(subG, HOME+"\\Communities\\" +
                              dominating_genre+"-"+str(genre_scores[dominating_genre])+"\\")

        with open(HOME+"\\Communities\\" +
                  dominating_genre+"-"+str(genre_scores[dominating_genre])+"\\statistics.csv", "w+") as file:
            file.write("Number of nodes,Number of Edges\n")
            file.write(str(len(subG.nodes())) + "," + str(len(subG.edges())))

        plot_graph(subG, HOME+"\\Communities\\" +
                   dominating_genre+"-"+str(genre_scores[dominating_genre])+"\\degree_dist.png")
    return genre_scores

# endregion

# region Centrality


"""
Ended up no using this metric either, as I was unable to translate what 
the values meant well enough.
Calculated average betweenness centrality for node in each genre
"""


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

# region Random network


"""
Takes a graph G (which is assumbed to be unweighted and undirected) and shuffles it using
NetworkX double_edge_swap, and swaps 20,000 to get nice and random.
Also takes in a number of times to do this -> iterations
"""


def random_network_gen(iterations, G):
    for i in range(iterations):
        # Makes a copy since the algorithm works in-place
        graph = nx.Graph.copy(G)

        # shuffle
        nx.algorithms.swap.double_edge_swap(
            graph, 20000, max_tries=30000)

        # computer clustering since we are here
        c = nx.clustering(graph)

        # Associate each node with its genre for later computation
        coeffs_and_genres = [(c[n], graph.nodes()[n]['genre']) for n in c]

        # output for later computation
        Functions.write_graph(graph, HOME+"\\RandomNetworks\\" +
                              str(i)+"\\")
        # My write_graph function only works for digraphs
        nx.write_edgelist(graph, HOME+"\\RandomNetworks\\" +
                          str(i)+"\\edges.csv", delimiter=',')

        # Save clustering coeffs for later
        with open(HOME+"\\RandomNetworks\\"+str(i)+"\\clusters.csv", "w+") as out:
            for a in coeffs_and_genres:
                out.write(str(a)+",")


"""
Computes average shortest path length for every random graph generated

Saved the values and a plotted graph to the graphs folder
"""


def pathing_randos(number):
    path = HOME + "\\RandomNetworks\\"
    for i in range(number):
        g = Functions.load_graph(path+str(i)+"\\", False, False)

        # Returns list of tuples
        path_lengths = nx.shortest_path_length(g)

        # So extract only the path length values
        lengths = []
        for tuple in path_lengths:
            for target in tuple[1]:
                lengths.append(int(tuple[1][target]))

        # Compute frequency, inspired to stack overflow link that is found in the
        # degree region of this file
        num = max(lengths)+1
        freq = [0 for d in range(num)]
        for d in lengths:
            freq[d] += 1

        with open(path+str(i)+"\\length_distribution.csv", "w+") as out:
            for f in freq:
                out.write(str(f)+',')

        plt.plot(range(num), freq)
        plt.xlabel('Path length')
        plt.ylabel('Frequency')
        plt.savefig(path+str(i)+"\\length_distribution.png")
        plt.close()


"""
Computes and plots the Average Next Neighbour for nodes of degree k, Knn(k),
for the random graphs and plots the distribution.
Saves file to RandomNetworks folder
"""


def average_next_neighbour(number):
    path = HOME + "\\RandomNetworks\\"
    average_overall = [0]*70
    for i in range(number):
        g = Functions.load_graph(path+str(i)+"\\", False, False)

        # Man I love Networkx
        annd = nx.average_neighbor_degree(g)

        degrees = [tup[1] for tup in g.degree()]

        dmax = max(degrees)+1

        # stats for nodes of degree k are at degree_k_sum[k]
        degree_k_sum = [0]*dmax
        degree_k_counters = [0]*dmax

        for node in annd:
            node_degree = g.degree()[node]  # Get k
            # Add the average neighbour degree for this node
            degree_k_sum[node_degree] += annd[node]
            # count number of nodes of degree k
            degree_k_counters[node_degree] += 1

        # Normalize every value by the number of nodes of degree k
        annd_for_k = [0]*dmax
        for j in range(len(degree_k_counters)):
            try:
                annd_for_k[j] = degree_k_sum[j]/degree_k_counters[j]
            except:
                # case where there are no nodes of degree j
                annd_for_k[j] = 0

        # add to distribution for all random networks
        for j in range(len(annd_for_k)):
            average_overall[j] += annd_for_k[j]

    x = range(dmax)[1:]

    # normalize
    for i in range(len(average_overall)):
        average_overall[i] = average_overall[i]/number

    y = average_overall[1:len(x)+1]

    plt.loglog(x, y)
    plt.xlabel("k")
    plt.ylabel("Knn(k)")

    plt.savefig(HOME+"\\RandomNetworks\\ANND.png")

    plt.close()


"""
Simply loads each random graphs clustering coeffs computed before and calulates
average by genre and returns a list of averages, where each value
is the average clustering coefficient for the entire graph
"""


def clustering_randoms(number):
    path = HOME + "\\RandomNetworks\\"
    averages = []
    for i in range(number):
        sum = 0
        with open(path+str(i)+"\\clusters.csv", 'r') as file:
            tuples = file.readline()
            tuples = tuples.split(',')[:-2:2]

            for tup in tuples:
                tup = tup[1:]
                sum += float(tup)

            average = sum / len(tuples)
            averages.append(average)
    return np.average(averages)


"""
Calculates standard deviation for a path length distribution generated by
pathing_randos()
length[i] = # of paths of length i
"""


def avg_stdv(lengths):
    sum = 0
    for i in range(len(lengths)):
        sum += (i+1)*lengths[i]
    mu = sum / np.sum(lengths)
    N = np.sum(lengths)

    numerator = 0
    for i in range(1, len(lengths)):
        dev = (i - mu)**2
        numerator += dev * lengths[i]

    stddev = np.sqrt((numerator)/N)

    return mu, stddev


"""
Calculate the standard deviation for across ALL ditributions
"""


def path_distribution(number):
    path = HOME + "\\RandomNetworks\\"
    path_lengths = [0]*30
    for i in range(number):

        with open(path+str(i)+"\\length_distribution.csv", "r") as file:
            # Get the distribution as integers
            content = file.readline().split(',')[:-1]
            distribution = np.array(content).astype(int)
            # Add this graphs distribution to the total.
            for i in range(len(distribution)):
                path_lengths[i] += distribution[i]

    return avg_stdv(path_lengths)


"""
Calculates all EI indices for all random graphs, or if this was already
done, loads them from the files that should exist after 1 run of this function.

This takes about ~30 minutes per graph of 10,000+ edges, so please mind the CPU heat.

returns the average by genre, across every graph
"""


def ei_indicies_randoms(number, genres, load_from_file=False):
    path = HOME + "\\RandomNetworks\\"
    all_averages = {}
    if load_from_file:
        for i in range(number):
            with open(path+str(i)+"\\ei_indicies_avg.csv") as file:
                lines = file.readlines()[1:]
                for lines in lines:
                    words = lines.split(',')
                    average = float(words[1])
                    try:
                        all_averages[words[0]] += average
                    except:
                        all_averages[words[0]] = average
    else:
        for i in range(number):

            g = Functions.load_graph(path+str(i)+"\\", False, False)

            g_indicies = ei_index(g, genres)

            # save it, this stuff was expensive
            write_genre_by_genre(
                g_indicies, "\\RandomNetworks\\"+str(i)+"\\ei_indicies.csv")

            g_indicies_averages = {}

            # For each genre A
            for A in g_indicies:
                sum_for_A = 0
                # Add the values up for this genre
                for B in g_indicies[A]:
                    sum_for_A += g_indicies[A][B]

                # and get the average
                average_for_A = sum_for_A/len(g_indicies[A])

                # set it for this graph to save later
                g_indicies_averages[A] = average_for_A

                # sum it up for average across all graphs
                try:
                    all_averages[A] += average_for_A
                except:
                    all_averages[A] = average_for_A

            # Save it again as Im computationally poor
            write_by_genre(genres,  "\\RandomNetworks\\"+str(i)+"\\ei_indicies_avg.csv",
                           g_indicies_averages, "Average EI index")

    # compute averages and return
    for genre in all_averages:
        all_averages[genre] = all_averages[genre] / 100
    return all_averages


"""
Driver Method
"""


def comparison_to_random_networks(G, genres):

    amount = 100
    ei_avgs = ei_indicies_randoms(amount, genres, True)
    print("Average EI Index for all random networks by genre:")
    for genre in ei_avgs:
        print("Genre: "+genre+"\t\t\t \u03BC: "+str(ei_avgs[genre]))
    print("\nOverall Random EI \u03BC: " +
          str(np.average(list(ei_avgs.values()))))

    print()

    mu, stddev = path_distribution(amount)
    print("Average and Standard Deviation of path length andom networks:")
    print("\u03BC = "+str(mu))
    print("\u03C3 = "+str(stddev))

    with open(HOME+"\\AnalysisData\\length_distribution.csv", "r") as file:
        content = file.readline().split(',')[:-1]
        distribution = np.array(content).astype(int)
        path_lengths = [0]*len(distribution)
        for i in range(len(distribution)):
            path_lengths[i] += distribution[i]
    mu, stddev = avg_stdv(path_lengths)
    print("Average and Standard Deviation of path length for our network")
    print("\u03BC = "+str(mu))
    print("\u03C3 = "+str(stddev))

    print("\nAverage clustering co-eff for 100 random networks: " +
          str(clustering_randoms(amount)))

    c = nx.clustering(G)

    clusters = [c[n] for n in c]

    print("\nAverage clustering of our network: " + str(np.average(clusters)))

    print()
    average_next_neighbour(amount)
    print("Knn(k) has been graphed in the random networks folder")

# endregion


"""
Output a dictionary where the keys are the values in 'genres' to the specified path
and give the column the specified name.
"""


def write_by_genre(genres, path, dict, name):

    if os.path.exists(HOME+path):
        os.remove(HOME+path)

    with open(HOME + path, "w+") as outfile:
        outfile.write(
            "Genre,"+name+"\n")
        for genre in genres:
            outfile.write(genre+","+str(dict[genre]) + "\n")


"""
outputs a dictionary of dictionries as csv to a path
"""


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


"""
Driver method

All computation done are commented out, but kept in line so all you have to do is
ensure the proper files exist and uncomment the method

Note: Not all things that were computed were used in the report.
"""
if __name__ == "__main__":
    relative_path = "\\Level2\\Finalized Graph\\"

    graph = Functions.load_graph(HOME+relative_path)
    unweighted_graph = Functions.load_graph(HOME+relative_path, False, False)
    inverted_graph = Functions.load_graph(HOME+relative_path, True)
    undirected_graph = graph.to_undirected()
    undirected_unweighted_graph = unweighted_graph.to_undirected()
    genres = Functions.load_genres_from_file()

    """
    comparison_to_random_networks(undirected_unweighted_graph, genres)
    degree_dist_all(graph)
    average_next_neighbour()
    
    
    indicies = ei_index(undirected_graph, genres)
    write_genre_by_genre(indicies,
                         "\\AnalysisData\\ei_indicies.csv")
    avg_indegrees, max_degrees, cluster_coefficients = degree_analysis(
        graph, genres)
    write_by_genre(genres, "\\AnalysisData\\in_degrees.csv",
                   avg_indegrees, "Average In-Degree")
    write_by_genre(genres, "\\AnalysisData\\max_degrees.csv",
                   max_degrees, "Highest In-Degree")
    write_by_genre(genres, "\\AnalysisData\\clustering_coefficient.csv",
                   cluster_coefficients, "Average Clustering Coefficient")

    coeffs = clustering_by_sub_graph(graph, genres)
    write_by_genre(genres, "\\AnalysisData\\clustering_coefficient_subgraphs.csv",
                   coeffs, "Average Clustering Coefficient")
                   
                   
    community_scores = greedy_modularity(undirected_graph, graph)
    write_by_genre(genres, "\\AnalysisData\\community_scores.csv",
                   community_scores, "Communities Dominated")
                   
    enter_hops = fewest_hops_to_enter(inverted_graph, genres)
    write_by_genre(genres, "\\AnalysisData\\hops_to_enter.csv",
                   enter_hops, "Average Fewest Hops To Enter")
    
    
    exit_hops = fewest_hops_to_exit(inverted_graph, genres)
    write_by_genre(genres, "\\AnalysisData\\hops_to_exit.csv",
                   exit_hops, "Average Fewest Hops To Exit")
    
    

    avg_dist_btwn_genres = from_one_to_the_other(
        inverted_graph, genres)
    write_genre_by_genre(avg_dist_btwn_genres,
                         "\\AnalysisData\\distance.csv")    

    probs = probability_analysis(graph)
    write_genre_by_genre(probs, "\\AnalysisData\\genre_probabilities.csv")
    
    num_of_nodes = count_nodes(graph, genres)
    write_by_genre(genres, "\\AnalysisData\\node_count.csv",num_of_nodes, "Number of Nodes")
    
    between = betweenness(inverted_graph, genres)
    write_by_genre(genres, "\\AnalysisData\\betweeness.csv",
                   between, "Average Betweeness")
    
    """
