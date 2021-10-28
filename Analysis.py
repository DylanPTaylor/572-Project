from itertools import count
from datetime import datetime
import Functions
import Youtube
import json
import networkx as nx
import os

HOME = Functions.Home()


def load_graph(path):
    graph = nx.DiGraph()

    with open(HOME+path) as node_file:
        node = node_file.readline()
        node = node_file.readline()
        while (node != ""):
            node = node.split(';')
            node[2] = node[2].strip()

            graph.add_node(node[0], genre=node[1], probabilities=node[2])
            node = node_file.readline()
    return graph


def crunch_probabilities(nodes):
    # Expect nodes as a dictionary of dictionaries, top level keys are video ids
    #counter = 0
    # This is also a dictionary of dictionaries, keys are genre names, values are dictionaries of the probability of
    # seeing each genre
    genre_probability_sets = {}
    genre_counters = {}
    for node in nodes:
        this_nodes_probabilities = json.loads(
            nodes[node]["probabilities"].replace("\'", "\""))
        this_nodes_genre = nodes[node]["genre"]

        # Error checking: skip ones who dont sum to zero
        sum = 0
        for prob in this_nodes_probabilities:
            sum += float(this_nodes_probabilities[prob])

        if sum > 1.1 or sum < 0.9:
            # print(str(counter)+":"+str(sum))
            #counter += 1
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


def write_file(probs, path):
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


if __name__ == "__main__":

    graph = load_graph("\\Level3\\28-10-2021\\nodes.csv")

    probs = crunch_probabilities(dict(graph.nodes()))

    translate(probs)

    write_file(probs, "\\genre_probabilities-" +
               datetime.now().strftime("%d-%m-%Y")+".csv")
