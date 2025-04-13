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

limit = 10

#print(api.user("mrekk").rank_history.data[0])

class Player:
    def __init__(self, id):
        self.id = id
        self.username = api.user(id).username
        self.scores = api.user_scores(self.id, "best", mode="osu", limit=100)
    
    def DisplayTopPlays(self):
        for i in range(len(self.scores)):
            print(f"{self.scores[i].beatmapset.artist:<40.38} {self.scores[i].beatmapset.title:<40.38} {self.scores[i].pp}")
        
player0 = Player(7562902)
#print(player0.id, player0.username)
#player0.DisplayTopPlays()

def GeneratePlayers(n_players): # generate players from rank 1 to n
    PlayerList = []
    for i in range(n_players):
        player_id = ranking[i].user.id
        player = Player(player_id)
        PlayerList.append(player)

    return PlayerList

PlayerList = GeneratePlayers(2)

PlayerList[2].DisplayTopPlays()

#scores_mrekk = (api.user_scores(ranking[0].user.id, "best", mode="osu", limit=limit))

'''
for i in range(limit):
    print(scores_mrekk[i].pp)
'''

"""while True:
    beatmap_id = input("beatmap id: ")
    beatmap = api.beatmap_attributes(beatmap_id,mods="DT")
    print(beatmap.attributes.aim_difficulty, beatmap.attributes.speed_difficulty)"""