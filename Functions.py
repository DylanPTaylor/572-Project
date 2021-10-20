import json
import Youtube
import random
import os
import networkx as nx
from datetime import datetime
import shutil


def Home():
    return "C:\\Users\\dylan\\OneDrive\\Desktop\\School\\Code\\Python\\572"


HOME = Home()


def Log(message):
    with open(HOME+"\\CrawlerData\\log.txt", 'a') as log:
        log.write("\nDate Time: " +
                  datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'\n')
        log.write("Message: "+message+'\n')
        log.close()

# region Graph Functions


def graph_data(crawler):
    populate_crawler(crawler)
    update_node(crawler)
    if not crawler.graph.has_node(crawler.video):
        add_self(crawler)


def compute_averages(graph):
    for Id in graph.nodes():
        node = graph.nodes()[Id]
        probabilities = node["probabilities"]

        in_degree = graph.in_degree(Id, "weight")

        try:
            for probability in probabilities:
                average = float(probabilities[probability]) / in_degree
                probabilities[probability] = float(int(average*10000)/10000)
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
                probabilities[probability] = float(int(average*10000)/10000)
            graph.nodes()[Id]["probablities"] = probabilities
        except Exception:
            continue


def update_node(crawler):
    probabilities = {}

    genreProbability = 1 / crawler.numberOfCandidatesToConsider

    # Get the probability of choosing each genre from these candidates
    for candidate in crawler.candidateVideos:
        candidateGenre = crawler.candidateGenres[candidate]

        try:
            probabilities[candidateGenre] += genreProbability
        except KeyError:
            probabilities[candidateGenre] = genreProbability

    # Add those probabilities to this nodes probability array
    for genre in probabilities.keys():
        if genre in crawler.probabilities:
            crawler.probabilities[genre] += probabilities[genre]
        else:
            crawler.probabilities[genre] = probabilities[genre]

    node = crawler.graph.nodes()[crawler.video]
    node["probabilities"] = crawler.probabilities


def sum_probabilities(initial, toAdd):
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


def add_self(crawler):
    crawler.graph.add_node(crawler.video, genre=crawler.genre,
                           probabilities=crawler.probabilities)


def write_nodes(nodes, path):
    with open(path, 'w+') as file:
        file.write("Id;Genre;Probabilities\n")
        for id in nodes:
            node = nodes[id]
            file.write(
                id + ";"+str(node["genre"])+"; "+str(node["probabilities"])+"\n")


def write_edges(adjLists, path):
    with open(path, 'w+') as file:
        file.write("Source,Target,{weight}\n")
        for thisNode in adjLists:
            for neighbour in adjLists[thisNode]:
                file.write(
                    thisNode+','+neighbour+',{' + str(adjLists[thisNode][neighbour]['weight']) + '}\n')


def write_graph(graph, path):
    if not os.path.exists(path):
        os.makedirs(path)
    write_edges(dict(graph.adj), path+"edges.csv")
    write_nodes(graph.nodes(), path+"nodes.csv")

# right now edge weights of final graph are number of crawlers to traverse that edge,
# not how many times each crawler travsersed it, summed up.


def compile_crawlers(path):
    final_graph = nx.DiGraph()

    final_graph_path = path+"\\CompiledGraphs\\"+datetime.now().strftime("%d-%m-%Y")
    crawlers_path = path+"\\CrawlerData\\"

    if os.path.exists(final_graph_path):
        shutil.rmtree(final_graph_path)

    os.makedirs(final_graph_path)

    for crawler in os.listdir(crawlers_path):
        if crawler == "log.txt":
            continue

        nodes = open(crawlers_path+crawler+"\\nodes.csv", 'r')
        nodes.readline()
        node = nodes.readline()
        while (node != ""):
            node = node.replace("\n", "").split(';')

            # If this node is already in this graph, add this duplicated nodes probabilities to it,
            # will take average later using the duplication field
            if final_graph.has_node(node[0]):
                nodeInGraph = final_graph.nodes()[node[0]]
                nodeInGraph["probabilities"] = sum_probabilities(
                    nodeInGraph["probabilities"], node[2])
                nodeInGraph["duplication"] = int(
                    nodeInGraph["duplication"]) + 1
            # other wise add it with this nodes probabilities.
            else:
                final_graph.add_node(
                    node[0], genre=node[1], probabilities=node[2], duplication=1)

            node = nodes.readline()

        edges = open(crawlers_path+crawler+"\\edges.csv", 'r')
        edges.readline()
        edge = edges.readline()
        while (edge != ''):
            edge = edge.split(',')

            try:
                edge_weight = final_graph.get_edge_data(
                    edge[0], edge[1], default=0)
                final_graph[edge[0]][edge[1]]["weight"] = int(
                    edge_weight["weight"])+1
            except Exception:
                final_graph.add_edge(edge[0], edge[1], weight=1)

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
    json_data = Youtube.fetch_data(crawler.video)

    crawler.candidateVideos = json_data["candidate videos"][:
                                                            crawler.numberOfCandidatesToConsider]
    crawler.genre = json_data["genre"]
    crawler.candidateGenres = Youtube.get_genres(crawler.candidateVideos)
    if crawler.graph.has_node(crawler.video):
        node = crawler.graph.nodes()[crawler.video]
        crawler.probabilities = node["probabilities"]
    else:
        crawler.graph.add_node(
            crawler.video, genre=crawler.genre, probabilities={})
        crawler.probabilities = {}

    json_data = None

# endregion
