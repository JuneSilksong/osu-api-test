import os
from dotenv import load_dotenv
from ossapi import Ossapi
import pandas as pd
import numpy as np

load_dotenv("api.env")

client_id = os.getenv("client_id")
client_secret = os.getenv("client_secret")

api = Ossapi(client_id, client_secret)
ranking = api.ranking("osu", "performance").ranking

limit = 12

class Player:
    def __init__(self, id):
        self.id = id
        self.username = api.user(id).username
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

    return PlayerList

n_players = 100

PlayerList = GeneratePlayers(n_players)

def GetPlayerScoreModList(scores):
    modList = []
    for score in scores:
        mods = ""
        for mod in score.mods:
            if mod.acronym == "CL":
                mods += "NM"
                break
            mods += mod.acronym
        modList.append(mods)
    return modList

def ScoreRating(beatmap_id, mods):
    beatmap = api.beatmap_attributes(beatmap_id,mods=mods)
    beatmapScore = (beatmap.attributes.aim_difficulty - beatmap.attributes.speed_difficulty) * beatmap.attributes.star_rating
    return beatmapScore

def GetPlayerArchetype(Player):
    modList = GetPlayerScoreModList(Player.scores)
    #print(modList)
    playerScore = 0
    for i in range(limit):
        #print(Player.scores[i].beatmap.id)
        mapScore = ScoreRating(Player.scores[i].beatmap.id, modList[i])
        playerScore += mapScore
    return playerScore

for player in PlayerList:
    playerScore = GetPlayerArchetype(player)
    print(player.username, playerScore)