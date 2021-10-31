def Home():
    return "C:\\Users\\dylan\\OneDrive\\Desktop\\School\\Code\\Python\\572"

import json
from Youtube import fetch_data, get_genres
import random
import os
import networkx as nx
from datetime import datetime
import shutil



HOME = Home()


def Log(message):
    with open(HOME+"\\log.txt", 'a') as log:
        log.write("\n---Date Time: " +
                  datetime.now().strftime("%d/%m/%Y %H:%M:%S")+' ---')
        log.write("Message: "+message+'\n')
        log.close()

def load_genres_from_file():
    genres = []
    with open(HOME+"\\VideoData\\genres.json") as in_file:
        genre_list = json.load(in_file)
        for dict in genre_list:
            genres.append(dict["title"])
    
    return genres

# region Graph Functions


def graph_data(crawler):
    populate_crawler(crawler)
    update_node(crawler)


def compute_averages(graph):
    for Id in graph.nodes():
        node = graph.nodes()[Id]
        probabilities = node["probabilities"]

        in_degree = graph.in_degree(Id)

        sum = 0
        for prob in probabilities:
            sum += float(probabilities[prob])

        if in_degree == 0: in_degree+=1

        try:
            if sum/in_degree > 1.1 or sum/in_degree < 0.9:
                    in_degree += 1
            for probability in probabilities:
                average = float(probabilities[probability]) / in_degree
                #probabilities[probability] = float(int(average*10000)/10000) #Might be sending some probs to zero
                probabilities[probability] = average
            graph.nodes()[Id]["probablities"] = probabilities
        except Exception:
            continue


def compute_final_averages(graph):
    for Id in graph.nodes():
        node = graph.nodes()[Id]
        probabilities = node["probabilities"]

        duplication = node["duplication"]

        try:
            for probability in probabilities:
                average = float(probabilities[probability]) / duplication
                probabilities[probability] = average
            graph.nodes()[Id]["probablities"] = probabilities
        except Exception:
            continue


def update_node(crawler):
    probabilities = {}

    genreProbability = 1 / len(crawler.candidateVideos)

    # Get the probability of choosing each genre from these candidates
    for candidate in crawler.candidateVideos:
        candidateGenre = crawler.candidateGenres[candidate]
        try:
            probabilities[candidateGenre] += genreProbability
        except KeyError:
            probabilities[candidateGenre] = genreProbability

    # Add those probabilities to this nodes probability array
    for genre in probabilities:
        try:
            crawler.probabilities[genre] += probabilities[genre]
        except KeyError:
            crawler.probabilities[genre] = probabilities[genre]

    node = crawler.graph.nodes()[crawler.video]
    node["probabilities"] = crawler.probabilities


def add_probabilities(initial, toAdd):
    if type(initial) != dict:
        initial = json.loads(initial.strip().replace("\'", "\""))
    if type(toAdd) != dict:
        toAdd = json.loads(toAdd.strip().replace("\'", "\""))
    for genre in toAdd:
        try:
            initial[genre] += toAdd[genre]
        except KeyError:
            initial[genre] = toAdd[genre]
    return initial


def write_nodes(nodes, path):
    with open(path, 'w+') as file:
        file.write("Id;Genre;Probabilities\n")
        for id in nodes:
            node = nodes[id]
            file.write(
                id + ";"+str(node["genre"])+"; "+str(node["probabilities"])+"\n")


def write_edges(adjLists, path):
    with open(path, 'w+') as file:
        file.write("Source,Target,weight\n")
        for thisNode in adjLists:
            for neighbour in adjLists[thisNode]:
                file.write(
                    thisNode+','+neighbour+',' + str(adjLists[thisNode][neighbour]['weight']) + '\n')

def load_edges(graph,path):
    edges_path = path+"edges.csv"
    with open(edges_path) as edges_file:
        edge = edges_file.readline()  # header
        edge = edges_file.readline()  # first edge
        while (edge != ""):
            edge = edge.split(",")
            graph.add_edge(edge[0], edge[1], weight=int(edge[2]))
            edge = edges_file.readline()
def load_nodes(graph, path):
    nodes_path = path + "nodes.csv"
    with open(nodes_path) as node_file:
        node = node_file.readline()  # header
        node = node_file.readline()  # first node
        while (node != ""):
            node = node.split(';')
            node[2] = node[2].strip()

            graph.add_node(node[0], genre=node[1], probabilities=json.loads(
                node[2].replace("\'", "\"")))
            node = node_file.readline()

def load_graph(path):
    graph = nx.DiGraph()

    load_nodes(graph, path)
    load_edges(graph, path)
    
    return graph
    
def write_graph(graph, path):
    if not os.path.exists(path):
        os.makedirs(path)
    write_edges(dict(graph.adj), path+"edges.csv")
    write_nodes(graph.nodes(), path+"nodes.csv")

#  probabilities are a little weird soemtimes
def compile_networks(source, target):
    final_graph = nx.DiGraph()

    final_graph_path = target+datetime.now().strftime("%d-%m-%Y")

    if os.path.exists(final_graph_path):
        shutil.rmtree(final_graph_path)

    os.makedirs(final_graph_path)

    for graph in os.listdir(source):

        nodes = open(source+graph+"\\nodes.csv", 'r')
        nodes.readline()
        node = nodes.readline()
        while (node != ""):
            node = node.replace("\n", "").split(';')

            # If this node is already in this graph, add this duplicated nodes probabilities to it,
            # will take average later using the duplication field
            if final_graph.has_node(node[0]):
                nodeInGraph = final_graph.nodes()[node[0]]
                nodeInGraph["probabilities"] = add_probabilities(
                    nodeInGraph["probabilities"], node[2])
                nodeInGraph["duplication"] = int(
                    nodeInGraph["duplication"]) + 1
            # other wise add it with this nodes probabilities.
            else:
                final_graph.add_node(
                    node[0], genre=node[1], probabilities=node[2], duplication=1)

            node = nodes.readline()

        edges = open(source+graph+"\\edges.csv", 'r')
        edges.readline()
        edge = edges.readline()
        while (edge != ''):
            edge = edge.split(',')

            edge_weight = final_graph.get_edge_data(
                edge[0], edge[1], default=0)
            if edge_weight == 0:
                final_graph.add_edge(edge[0], edge[1], weight=int(edge[2]))
            else:
                final_graph[edge[0]][edge[1]]["weight"] = int(
                    edge_weight["weight"])+int(edge[2])

            edge = edges.readline()

    compute_final_averages(final_graph)
    write_graph(final_graph, final_graph_path+"\\")
# endregion

# region Crawler Functions


def select_random(videos, numberToPick):
    random_videos = []
    for i in range(numberToPick):
        my_random_int = random.randint(0, len(videos)-1)
        selected_video = videos[my_random_int]

        videos.remove(selected_video)
        random_videos.append(selected_video)
    return random_videos


def populate_crawler(crawler):
    json_data = fetch_data(crawler.video)

    crawler.candidateVideos = json_data["candidate videos"][:
                                                            crawler.numberOfCandidatesToConsider]
    crawler.genre = json_data["genre"]
    crawler.candidateGenres = get_genres(crawler.candidateVideos)
    if crawler.graph.has_node(crawler.video):
        node = crawler.graph.nodes()[crawler.video]
        crawler.probabilities = node["probabilities"]
    else:
        crawler.graph.add_node(
            crawler.video, genre=crawler.genre, probabilities={})
        crawler.probabilities = {}

    json_data = None


def add_self(crawler):
    crawler.graph.add_node(crawler.video, genre=crawler.genre,
                           probabilities=crawler.probabilities)
# endregion
