"""
Author: Dylan Taylor
UCID: 30078900
Date: 08/12/2021

This class was created entirely by the Author an did not utilize any references
outside of the typical syntax-related google searches

If you wish to start multiple crawlers on the same IP address, list them (and their genre
for loggin reasons) if the starting_videos dict below as "id":"genre" pairs

otherwise, specify only one video, run it on the IP, and then switch IPs with a VPN.

To improve this code I recommend using CMD-line commands for your vpn to switch it
between spawning crawlers, or using the TOR network (this will be much more anonymous at the expens of time) 
"""
import Crawler
import time
import Functions
from datetime import datetime

HOME = Functions.Home()

# 3 at 6^3 ~= 3,200 api calls
starting_videos = {}

# default parameters
ttl = 6
candidates_to_watch = 3
candidatesToConsider = 10

# for logging
nodes = 0
edges = 0


def compile():
    Functions.Log("Compiling crawler networks...")
    Functions.compile_networks(HOME+"\\Level1\\", HOME+"\\Level2\\")
    Functions.Log("Finished compiling network.")


def run_crawler(video):
    global nodes
    global edges

    crawler = Crawler.Crawler(
        ttl, candidatesToConsider, candidates_to_watch, video)

    crawler.watch()

    Functions.compute_averages(crawler.graph)
    Functions.write_graph(crawler.graph, HOME +
                          "\\Level1\\"+crawler.video+"-"+datetime.now().strftime("%d-%m-%Y")+"\\")

    nodes += len(list(crawler.graph.nodes()))
    edges += len(list(crawler.graph.edges()))


def main():
    Functions.Log("Starting crawlers...")

    total_time = time.time()

    for video in starting_videos:
        start_time = time.time()
        try:
            Functions.Log("Starting crawler on "+video)
            run_crawler(video)
            Functions.Log("Crawler finished. \nVideo: "+video+"\nGenre:"+starting_videos[video]+"\nTTL: " + str(ttl) + ", Branches: "+str(candidates_to_watch) +
                          "\n Number of Nodes: "+str(nodes)+"\n Number of Edges: "+str(edges)+"\nExecution time: "+str(((time.time() - start_time)//60))+" minutes, " +
                          str(((time.time() - start_time) % 60))+" seconds")
        except Exception as e:
            Functions.Log("Crawler failed.  \nVideo: " +
                          video+"\nException: "+str(e))

    Functions.Log("All Crawlers executed. Total execution time: "+str(((time.time() - total_time)//60))+" minutes, " +
                  str(((time.time() - total_time) % 60))+" seconds")


if __name__ == "__main__":
    main()
    compile()
