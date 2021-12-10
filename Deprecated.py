"""
Author: Dylan Taylor
Date: 08/12/2021
Dont look at this its hideous and useless (relatable right?)
"""

import json
import Youtube
import os
import networkx as nx


# region Functions
GENRE_WEIGHT = 10
TAG_WEIGHT = 1
"""
    if from_file:
        if not os.path.exists(HOME+"\\VideoData\\html\\"+crawler.video+".json"):
            Youtube.get_html_data(crawler.video)
        if not os.path.exists(HOME+"\\VideoData\\api\\"+crawler.video+".json"):
            Youtube.get_api_data(crawler.video)

        with open(HOME+"\\VideoData\\html\\"+crawler.video+".json") as file:
            json_data = json.load(file)
            file.close()
    else:
"""


# region Pick by attribute


def get_hamming_weights(crawler):
    weights = {}
    for candidate in crawler.candidateVideos:
        weights[candidate] = 0

        for tag in range(len(crawler.tags)):
            # if this candidate has this tag
            if (crawler.tags[tag] in crawler.candidateTags[candidate]):
                weights[candidate] += TAG_WEIGHT
    return weights


def get_hamming_weights_v2(crawler):
    weights = {}
    for candidate in crawler.candidateVideos:
        weights[candidate] = 0

        for tag in range(len(crawler.tags)):
            # if this candidate does not have this tag
            if (crawler.tags[tag] not in crawler.candidateTags[candidate]):
                weights[candidate] += TAG_WEIGHT
    return weights


def best_by_tags(crawler, numberToGet):
    hammingWeights = get_hamming_weights(crawler)

    impossbleWeight = len(crawler.tags) + 1

    bestCandidates = []
    for i in range(numberToGet):

        candidateWithLeastWeight = next(iter(hammingWeights))

        for candidate in hammingWeights.keys():
            candidateWeight = hammingWeights[candidate]

            if (candidateWeight < hammingWeights[candidateWithLeastWeight]):
                candidateWithLeastWeight = candidate

        # We found one of the best candidates, save it
        bestCandidates.append(candidateWithLeastWeight)
        # Set its weight to something we can guarentee will not be
        # the least weight again to avoid duplication
        hammingWeights[candidateWithLeastWeight] = impossbleWeight
    return bestCandidates


def best_by_genre(crawler, numberToGet):
    bestCandidates = []
    for i in range(numberToGet):
        # get the first video whose genre does not match the crawlers
        candidate = next((c for c in crawler.candidateVideos if (
            crawler.candidateGenres[c] != crawler.genre)), None)
        if candidate != None:
            bestCandidates.append(candidate)
            crawler.candidateVideos.remove(candidate)
        else:
            break
    return bestCandidates


def find_best_candidates_v1(crawler):
    numberToGet = crawler.branches
    # base the best ones by genre
    bestCandidates = best_by_genre(crawler, numberToGet)

    if len(bestCandidates) != numberToGet:
        # get the rest based on tag
        moreCandidates = best_by_tags(
            crawler, numberToGet - len(bestCandidates))
        for candidate in moreCandidates:
            bestCandidates.append(candidate)

    return bestCandidates


# SECOND ALGO


def compare_genres(crawler):
    candidateValues = {}
    for candidate in crawler.candidateVideos:
        # get the first video whose genre does not match the crawlers
        if crawler.candidateGenres[candidate] != crawler.genre:
            candidateValues[candidate] = GENRE_WEIGHT
        else:
            candidateValues[candidate] = 0
    return candidateValues

# What if there are no tags?


def compare_tags(crawler, values):
    weights = get_hamming_weights_v2(crawler)
    for candidate in crawler.candidateVideos:
        values[candidate] += weights[candidate]
    return values


def find_best_candidates_v2(crawler):
    numberToGet = crawler.branches

    candidateValues = compare_genres(crawler)

    candidateValues = compare_tags(crawler, candidateValues)

    bestCandidates = []
    for i in range(numberToGet):
        candidateWithHighestValue = next(iter(candidateValues))

        for candidate in candidateValues.keys():
            candidateValue = candidateValues[candidate]

            if (candidateValue > candidateValues[candidateWithHighestValue]):
                candidateWithHighestValue = candidate

        # We found one of the best candidates, save it
        bestCandidates.append(candidateWithHighestValue)
        candidateValues[candidateWithHighestValue] = -1

    return bestCandidates

# endregion
# endregion

# region Youtube


def get_api_data(video):
    if os.path.exists(HOME+"\\VideoData\\api\\"+video+".json"):
        return

    youtube = get_authenticated_service()

    # Get API data for this video
    request = youtube.videos().list(
        part="snippet",
        id=video,
        key=KEY
    )
    response = request.execute()

    items = response["items"]

    items = items[0]

    snippet = items["snippet"]

    try:
        tags = [tag.upper() for tag in snippet["tags"]]
    except KeyError:
        tags = []

    genreId = snippet["categoryId"]

    data = {"id": items["id"], "tags": tags, "genreId": genreId}

    with open(HOME+"\\VideoData\\api\\"+video+".json", 'w') as file:
        json.dump(data, file)


def get_html_data(video):
    if os.path.exists(HOME+"\\VideoData\\html\\"+video+".json"):
        return

    raw_html = requests.get(
        YOUTUBE_URI_BASE+YOUTUBE_WATCH_ARG+video)

    parser = BeautifulSoup(raw_html.text, 'html.parser')

    formatted_html = list(parser.children)[1]

    meta_data = formatted_html.head.find_all_next("meta")
    scripts = formatted_html.body.contents

    tags = parse_content(meta_data, 'keywords')
    tags = [tag.upper() for tag in tags.split(", ")]
    genre = parse_content(meta_data, 'genre')
    channel_id = parse_content(meta_data, 'channelId')

    ytInitialData = get_yt_data(scripts)

    json_data = json.loads(ytInitialData)

    suggested_videos = json_data['contents']['twoColumnWatchNextResults']['secondaryResults']['secondaryResults']['results'][1:]

    suggested_videos = extract_ids(suggested_videos)

    data = {"videoId": video, "tags": tags, "genre": genre, "channel_id": channel_id,
            "candidate videos": suggested_videos}

    with open(HOME+"\\VideoData\\html\\"+video+".json", "w") as file:
        json.dump(data, file)


def get_video_genre(video):
    # check if its in the html folder
    if os.path.exists(HOME+"\\VideoData\\html\\"+video+".json"):
        with open(HOME+"\\VideoData\\html\\"+video+".json") as file:
            json_data = json.load(file)
            file.close()
        return json_data["genre"]

    # Other wise get it from api folder or youtube.
    genreId = get_genre_id(video)

    # Check if we have this genre
    with open(HOME+"\\VideoData\\genres.json", "r") as file:
        genres = json.load(file)
        file.close()

    for genre in genres:
        if genreId == genre["genreId"]:
            return genre["title"]

    # if not, get it and return it
    get_genre_names([genreId])
    return get_video_genre(video)


def get_tags(video):
        # check if we have the information for this video
    if os.path.exists(HOME+"\\VideoData\\api\\"+video+".json"):
        with open(HOME+"\\VideoData\\api\\"+video+".json") as file:
            data = json.load(file)
            return data["tags"]
    elif os.path.exists(HOME+"\\VideoData\\html\\"+video+".json"):
        with open(HOME+"\\VideoData\\html\\"+video+".json") as file:
            data = json.load(file)
            return data["tags"]

    youtube = get_authenticated_service()

    # Get API data for this video
    request = youtube.videos().list(
        part="snippet",
        id=video,
        key=KEY
    )
    response = request.execute()

    try:
        tags = [(tag.upper())
                for tag in response["items"][0]["snippet"]["tags"]]
    except KeyError:
        tags = []

    return tags


def get_genre_id(video):
    # Check if its in the api folder
    if os.path.exists(HOME+"\\VideoData\\api\\"+video+".json"):
        with open(HOME+"\\VideoData\\api\\"+video+".json") as file:
            json_data = json.load(file)
            file.close()
        return json_data["genreId"]
    # get the api data to get the genre
    else:
        get_api_data(video)
        return get_genre_id(video)


def watch_v1(self):

    if (self.TTL >= 0):

        Functions.populate_crawler(self, from_file=True)

        self.graph.add_nodes_from([(self.video, dict(genre=self.genre))])

        for candidate in self.candidateVideos:
            self.candidateTags[candidate] = Youtube.get_tags(candidate)
            self.candidateGenres[candidate] = Youtube.get_genre(candidate)

        videosToWatch = Functions.find_best_candidates_v1(self)

        for v in videosToWatch:
            spawn = self.spawn_crawler(v)
            spawn.watch()
            self.graph.add_edge(self.video, v)


# endregion


def clustering_by_sub_graph(graph, genres):
    cluster_coefficients = {}
    nodes = graph.nodes()
    for genre in genres:

        nodes_in_this_genre = [n for n in nodes if nodes[n]["genre"] == genre]

        subG = nx.Graph(nx.subgraph(graph, nodes_in_this_genre))

        nodes_coefficients = nx.clustering(subG, None, 'weight')

        sum = 0
        number_of_nodes = len(nodes_coefficients)
        for node in nodes_coefficients:
            sum += nodes_coefficients[node]

        cluster_coefficients[genre] = (sum / number_of_nodes)

    return cluster_coefficients


# get multiple videos genres without checking our data

def get_genres(videos):
    youtube = get_authenticated_service()
    # Get API data for these videos
    request = youtube.videos().list(
        part="id,snippet",
        id=videos,
        key=KEY
    )

    response = request.execute()

    items = response["items"]

    videoGenres = {}
    for item in items:
        snippet = item["snippet"]
        id = item["id"]
        genre = snippet["categoryId"]
        videoGenres[id] = genre

    return videoGenres
