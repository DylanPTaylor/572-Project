import networkx as nx
import Youtube
import Functions
import os

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

            if var == 0:
                return

            videosToWatch = Functions.select_random(
                self.candidateVideos, self.branches)

            for v in videosToWatch:
                spawn = self.spawn_crawler(v)
                spawn.watch()

                # Add or set edge weight for this video to the one we just watched

                edge_weight = self.graph.get_edge_data(
                    self.video, v, default=0)
                if edge_weight == 0:
                    self.graph.add_edge(self.video, v, weight=1)
                else:
                    self.graph[self.video][v]["weight"] = int(
                        edge_weight["weight"])+1
        else:
            Functions.graph_data(self)
