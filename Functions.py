import json
import Youtube
import random
import os
import networkx as nx
from datetime import datetime


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


def compile_crawlers(path):
    final_graph = nx.DiGraph()

    # os.makedirs(path+"compiled\\"+datetime.now().strftime("%d-%m-%Y"))

    for folder in os.listdir(path):
        if folder == "compiled" or folder == "log.txt":
            continue

        edges = open(path+folder+"\\edges.csv", 'r')
        edges.readline()
        edge = edges.readline()
        while (edge != ''):
            edge = edge.split(',')
            final_graph.add_edge(
                edge[0], edge[1], weight=edge[2].replace("{", "").replace("}", "").replace("\n", ""))
            edge = edges.readline()

        nodes = open(path+folder+"\\nodes.csv", 'r')
        nodes.readline()
        node = nodes.readline()
        while (node != ""):
            node = node.replace("\n", "").split(';')
            final_graph.add_node(node[0], genre=node[1], probabilities=node[2])
            node = nodes.readline()

        write_graph(final_graph, path+"compiled\\" +
                    datetime.now().strftime("%d-%m-%Y")+"\\")
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
