from networkx.algorithms.cuts import normalized_cut_size
import Crawler
import networkx as nx
import os
import time
#import Youtube
import Functions

HOME = Functions.Home()

# comedy, education, People & Blogs, News & Politics, News & Politics (cons), News & Politics (lib)
starting_videos = {"9lFkNHn8dYQ": "Gaming",
                   "FbG6Vg_d4Mc": "News & Politics (liberal)", "PpyNKmdvGVo": "Entertainment", "4JzwhcstTEA": "News & Politics (conservative)"}

ttl = 5
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
                          "\\CrawlerData\\"+crawler.video+"\\")

    nodes = len(list(crawler.graph.nodes()))
    edges = len(list(crawler.graph.edges()))


def main():
    Functions.Log("Starting crawlers...")

    total_time = time.time()

    for video in starting_videos:
        start_time = time.time()
        run_crawler(video)
        Functions.Log("Starting video "+video+" from genre "+starting_videos[video]+" stats: TTL: " + str(ttl) + ", Branches: "+str(candidates_to_watch) +
                      ", Nodes: "+str(nodes)+", Edges: "+str(edges)+"\nExcution time: "+str(((time.time() - start_time)//60))+" minutes, " +
                      str(((time.time() - start_time) % 60))+" seconds")

    Functions.Log("Total execution time:"+str(((time.time() - total_time)//60))+" minutes, " +
                  str(((time.time() - total_time) % 60))+" seconds")


if __name__ == "__main__":
    Functions.compile_crawlers(HOME)
