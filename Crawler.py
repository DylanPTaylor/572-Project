import networkx as nx
import Youtube
import Functions
import os

HOME = Functions.Home()


class Crawler:
    # NOTE on Adjacency lists vs matricies:
    # Space: Lists: O(n+m) and matricies: O(n^2)
    # Checking for edges: O(n) for lists and O(1) for matricies
    # Adding/deleting nodes: O(n) for lists and O(n^2) for matricies
    # Both add edges in O(1)
    # Generally, lists are better for sparse graphs while matricies are better for dense graphs
    # So, we use lists for crawlers, and a matrix for the final network

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

    # endregion
    def spawn_crawler(self, video):
        spawn = Crawler(
            self.TTL-1, self.numberOfCandidatesToConsider, self.branches, video)
        return spawn

    def watch(self):
        if (self.TTL > 0):
            Functions.graph_data(self)

            videosToWatch = Functions.select_random(
                self.candidateVideos, self.branches)

            for v in videosToWatch:
                spawn = self.spawn_crawler(v)
                spawn.watch()

                # Add or set edge weight for this video to the one we just watched
                try:
                    edge_weight = self.graph.get_edge_data(
                        self.video, v, default=0)
                    self.graph[self.video][v]["weight"] = int(
                        edge_weight["weight"])+1
                except Exception:
                    self.graph.add_edge(self.video, v, weight=1)
        else:
            Functions.graph_data(self)
