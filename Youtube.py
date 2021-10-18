# Youtube API documentation: https://developers.google.com/youtube/v3/docs/?hl=en_US
# I Created a google cloud project to store the API Key which is needed to authenticate with Youtubes DATA Api V3

# need 3 libraries to run this:

# pip install requests
# pip install --upgrade google-api-python-client
# pip install --upgrade google-auth-oauthlib google-auth-httplib2


# guideCategory seem important
# videoCategory might be the tags

# using "list" operation as get
# must use "part" paramter in all get requests

# default of 10,000 quota, or requests, per day for api calls
# our api calls should cost 1 each

# https://github.com/youtube/api-samples
# video resources has "suggestions" field and "topic details" - > might be what we are looking for


# might not need these
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
import argparse
import re

import os
import requests
import json
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
from Functions import Home

HOME = Home()
YOUTUBE_URI_BASE = 'https://youtube.com'
YOUTUBE_WATCH_ARG = '/watch?v='
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'
KEY = "AIzaSyDqr_J5Aa8VXQdGRYCT2Kn6uRoCcKYoWMI"


# region File Writing


def get_genre_names(Ids):
    youtube = get_authenticated_service()

    # Get API data for this video
    request = youtube.videoCategories().list(
        part="snippet",
        id=Ids,
        key=KEY
    )

    response = request.execute()

    items = response["items"]

    genres = []
    for item in items:
        genreId = item["id"]
        title = item["snippet"]["title"]
        genre = {}
        genre["genreId"] = genreId
        genre["title"] = title
        genres.append(genre)

    with open(HOME+"\\VideoData\\genres.json", "r") as file:
        json_data = json.load(file)
        file.close()

    for genre in genres:
        if all((g["genreId"] != genre["genreId"]) for g in json_data):
            json_data.append(genre)

    with open(HOME+"\\VideoData\\genres.json", "w") as file:
        json.dump(json_data, file)
# endregion

# region Helpers


def get_authenticated_service():
    return build(API_SERVICE_NAME, API_VERSION, developerKey=KEY)


def parse_content(dataset, content):
    for result in dataset:
        if 'name' in result.attrs.keys():
            if result.attrs['name'] == content:
                return result.attrs['content']
        elif 'itemprop' in result.attrs.keys():
            if result.attrs['itemprop'] == content:
                return result.attrs['content']


def get_yt_data(scripts):
    for script in scripts:
        if script.name == 'script' and isinstance(script.next, str) and script.next.startswith("var ytInitialData"):
            return script.next[script.next.find('{'):len(script.next)-1]


def extract_ids(dataset):
    ids = []
    for item in dataset:
        if 'compactVideoRenderer' in item.keys():
            ids.append(item['compactVideoRenderer']['videoId'])
    return ids
# endregion

# region In Memory


def translate_genre_id_to_name(id):
    # Check if we have this genre
    with open(HOME+"\\VideoData\\genres.json", "r") as file:
        genres = json.load(file)
        file.close()

    for genre in genres:
        if id == genre["genreId"]:
            return genre["title"]

    # if not, get it and return it
    get_genre_names([id])
    return translate_genre_id_to_name(id)

# get HTML data


def fetch_data(video):
    raw_html = requests.get(
        YOUTUBE_URI_BASE+YOUTUBE_WATCH_ARG+video)

    parser = BeautifulSoup(raw_html.text, 'html.parser')

    formatted_html = list(parser.children)[1]

    meta_data = formatted_html.head.find_all_next("meta")
    scripts = formatted_html.body.contents

    ytInitialData = get_yt_data(scripts)

    json_data = json.loads(ytInitialData)

    suggested_videos = json_data['contents']['twoColumnWatchNextResults']['secondaryResults']['secondaryResults']['results'][1:]

    suggested_videos = extract_ids(suggested_videos)

    genre = parse_content(meta_data, 'genre')

    data = {"videoId": video, "genre": genre,
            "candidate videos": suggested_videos}

    return data


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


# endregion
