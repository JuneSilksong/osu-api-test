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
    
    def display_plays(self):
        for i in range(len(self.scores)):
            print(f"{self.scores[i].beatmapset.artist:<40.38} {self.scores[i].beatmapset.title:<40.38} {self.scores[i].pp}")

def generate_players(n_players): # generate players from rank 1 to n
    player_list = []
    for i in range(n_players):
        player_id = ranking[i].user.id
        player = Player(player_id)
        player_list.append(player)
        print(f"Generated player {player.user.username}")

    return player_list

def get_player_score_mod_list(scores):
    mod_list = []
    for score in scores:
        mods = "NM"
        for mod in score.mods:
            if mod.acronym == "CL":
                break
            mods += mod.acronym
        mod_list.append(mods)
    return mod_list

def modlist_str_to_int(score):
    mod_sum = 0
    for mod in score.mods:
        if mod.acronym == "HR":
            mod_sum += 1
        if (mod.acronym == "DT") or (mod.acronym == "NC"):
            mod_sum += 2
    return mod_sum

def is_cached(beatmap_id, mods):
    f = open('beatmaps.csv', 'r')
    beatmaps_csv = f.read()
    string_pos = beatmaps_csv.find(f"{beatmap_id},{mods},")
    if string_pos == -1:
        return False
    else:
        return True

def cache_beatmap(beatmap_id, mods, beatmap_score):
    with open("beatmaps.csv", "a") as f:
        f.write(f"{beatmap_id},{mods},{beatmap_score}\n")

def get_cached_score(beatmap_id, mods):
    f = open('beatmaps.csv', 'r')
    beatmaps_csv = f.read()
    beatmap_string = re.search(fr"{beatmap_id},{mods},(.*?)\n", beatmaps_csv).group(1)
    beatmap_score = float(beatmap_string.split("\n")[0])
    return beatmap_score

def get_score_rating(beatmap_id, mods):
    if is_cached(beatmap_id, mods) == False:
        beatmap = api.beatmap_attributes(beatmap_id,mods=mods)
        beatmap_score = (beatmap.attributes.aim_difficulty - beatmap.attributes.speed_difficulty) * beatmap.attributes.star_rating
        cache_beatmap(beatmap_id, mods, beatmap_score)
    else:
        beatmap_score = get_cached_score(beatmap_id, mods)
    return beatmap_score

def get_player_archetype(Player):
    mod_list = get_player_score_mod_list(Player.scores)
    score_list = []
    player_score = 0.0
    for i in range(limit):
        beatmap_score = get_score_rating(Player.scores[i].beatmap.id, mod_list[i])
        player_score += beatmap_score
        score_list.append(Player.scores[i].beatmap.id)
    return player_score, score_list

n_players = 5
player_list = generate_players(n_players)

with open("player_archetypes.csv", "w") as f:
    for player in player_list:
        player_score, score_list = get_player_archetype(player)
        f.write(f"{player.user.statistics.pp}, {player_score}")
        for score in score_list:
            f.write(f", {score}",)
        f.write("\n")