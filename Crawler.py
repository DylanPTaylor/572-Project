"""
Author: Dylan Taylor
UCID: 30078900
Date: 08/12/2021

This class was created entirely by the Author an did not utilize any references
outside of the typical syntax-related google searches

After a crawler is spawn with the appropriate attributes,
the .watch() method must be called. The crawler will then go off
and randomly watch videos that were suggest to it.

It records the videos it sees and edges it traverses into the "graph" attribute
It is up to the user to then save this graph once the crawler returns.

If you want to make it better, multi-thread it
"""

import networkx as nx
import Functions

HOME = Functions.Home()

class Crawler:

    # region Attributes
    # default values
    TTL = 6     # how many hopes before we stop
    branches = 3    # final size of network will be at most TTL^branches +1
    numberOfCandidatesToConsider = 20
    video = None
    tags = []
    genre = ""

    candidateVideos = []
    candidateTags = {}
    candidateGenres = {}

    probabilities = {}

    graph = nx.DiGraph()

    # endregion

    # region Contructors
    def __init__(self, startingVideo):
        self.video = str(startingVideo)

    def __init__(self, ttl, candidatesToConsider, candidatesToWatch, startingVideo):
        self.TTL = int(ttl)
        self.branches = int(candidatesToWatch)
        self.video = str(startingVideo)
        self.numberOfCandidatesToConsider = int(candidatesToConsider)

    def spawn_crawler(self, video):
        spawn = Crawler(
            self.TTL-1, self.numberOfCandidatesToConsider, self.branches, video)
        return spawn
    # endregion

    def watch(self):
        if (self.TTL > 0):
            var = Functions.graph_data(self)

            # If we error
            if var == 0:
                return

            # Select a subset of the candidate videos to recurse on
            # at random
            videosToWatch = Functions.select_random(
                self.candidateVideos, self.branches)

            for v in videosToWatch:
                # Recurse
                spawn = self.spawn_crawler(v)
                spawn.watch()

                # Add or set edge weight for this video to the one we just watched

                edge_weight = self.graph.get_edge_data(
                    self.video, v, default=0)
                # If this edge doesnt exist, add it
                if edge_weight == 0:
                    self.graph.add_edge(self.video, v, weight=1)
                # Otherwise increment its weight
                else:
                    self.graph[self.video][v]["weight"] = int(
                        edge_weight["weight"])+1
        else:
            Functions.graph_data(self)
