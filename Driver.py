from networkx.algorithms.cuts import normalized_cut_size
import Crawler
import networkx as nx
import os
import time
import Functions
from datetime import datetime

HOME = Functions.Home()

# comedy, education, People & Blogs, News & Politics, News & Politics (cons), News & Politics (lib)
starting_videos = {"bOuEJf8Dr_4": "Nonprofits & Activism",
                   "uYew7v8PyhA": "News & Politics",
                   "Cj76WFeRNX0": "News & Politics"}

ttl = 6
candidates_to_watch = 3
candidatesToConsider = 10

nodes = 0
edges = 0


def run_crawler(video):
    global nodes
    global edges

    crawler = Crawler.Crawler(
        ttl, candidatesToConsider, candidates_to_watch, video)

    crawler.watch()

    Functions.compute_averages(crawler.graph)
    Functions.write_graph(crawler.graph, HOME +
                          "\\Level1\\"+crawler.video+"-"+datetime.now().strftime("%d-%m-%Y")+"\\")

    nodes = len(list(crawler.graph.nodes()))
    edges = len(list(crawler.graph.edges()))


def main():
    Functions.Log("Starting crawlers...")

    total_time = time.time()

    for video in starting_videos:
        start_time = time.time()
        try:
            run_crawler(video)
            Functions.Log("Crawler finished. \nVideo: "+video+"\nGenre:"+starting_videos[video]+"\nTTL: " + str(ttl) + ", Branches: "+str(candidates_to_watch) +
                          "\n Number of Nodes: "+str(nodes)+"\n Number of Edges: "+str(edges)+"\nExecution time: "+str(((time.time() - start_time)//60))+" minutes, " +
                          str(((time.time() - start_time) % 60))+" seconds")
        except Exception as e:
            Functions.Log("Crawler failed.  \nVideo: "+video+"\nException: "+e)

    Functions.Log("All Crawlers executed. Total execution time: "+str(((time.time() - total_time)//60))+" minutes, " +
                  str(((time.time() - total_time) % 60))+" seconds")

    Functions.Log("Compiling crawler networks...")
    Functions.compile_networks(HOME+"\\Level1\\", HOME+"\\Level2\\")

    Functions.compile_networks(HOME+"\\Level2\\", HOME+"\\Level3\\")
    Functions.Log("Finished compiling network.")


if __name__ == "__main__":
    main()
