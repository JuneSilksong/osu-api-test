import os
from dotenv import load_dotenv
from ossapi import Ossapi
import pandas as pd
import numpy as np
import csv
import re

load_dotenv("api.env")

client_id = os.getenv("client_id")
client_secret = os.getenv("client_secret")

api = Ossapi(client_id, client_secret)
ranking = api.ranking("osu", "performance").ranking

limit = 12

class Player:
    def __init__(self, id):
        self.user = api.user(id)
        self.id = id
        self.scores = api.user_scores(self.id, "best", mode="osu", limit=limit)
    
    def DisplayTopPlays(self):
        for i in range(len(self.scores)):
            print(f"{self.scores[i].beatmapset.artist:<40.38} {self.scores[i].beatmapset.title:<40.38} {self.scores[i].pp}")
        
def GeneratePlayers(n_players): # generate players from rank 1 to n
    PlayerList = []
    for i in range(n_players):
        player_id = ranking[i].user.id
        player = Player(player_id)
        PlayerList.append(player)
        print(f"Generated player {player.user.username}")

    return PlayerList

def GetPlayerScoreModList(scores):
    modList = []
    for score in scores:
        mods = "NM"
        for mod in score.mods:
            if mod.acronym == "CL":
                break
            mods += mod.acronym
        modList.append(mods)
    return modList

def IsCached(beatmap_id, mods):
    f = open('beatmaps.csv', 'r')
    beatmapsCSV = f.read()
    stringPos = beatmapsCSV.find(f"{beatmap_id},{mods},")
    if stringPos == -1:
        return False
    else:
        return True

def CacheBeatmap(beatmap_id, mods, beatmapScore):
    with open("beatmaps.csv", "a") as f:
        f.write(f"{beatmap_id},{mods},{beatmapScore}\n")

def GetCachedScore(beatmap_id, mods):
    f = open('beatmaps.csv', 'r')
    beatmapsCSV = f.read()
    beatmapString = re.search(fr"{beatmap_id},{mods},(.*?)\n", beatmapsCSV).group(1)
    beatmapScore = float(beatmapString.split("\n")[0])
    return beatmapScore

def ScoreRating(beatmap_id, mods):
    if IsCached(beatmap_id, mods) == False:
        beatmap = api.beatmap_attributes(beatmap_id,mods=mods)
        beatmapScore = (beatmap.attributes.aim_difficulty - beatmap.attributes.speed_difficulty) * beatmap.attributes.star_rating
        CacheBeatmap(beatmap_id, mods, beatmapScore)
    else:
        beatmapScore = GetCachedScore(beatmap_id, mods)
    return beatmapScore

def GetPlayerArchetype(Player):
    modList = GetPlayerScoreModList(Player.scores)
    playerScore = 0.0
    for i in range(limit):
        mapScore = ScoreRating(Player.scores[i].beatmap.id, modList[i])
        playerScore += mapScore
    return playerScore

n_players = 50
PlayerList = GeneratePlayers(n_players)

with open("playerarchetypes.txt", "w") as f:
    for player in PlayerList:
        playerScore = GetPlayerArchetype(player)
        f.write(f"{player.user.username}, {playerScore}\n")