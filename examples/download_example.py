"""
Author: Yorai Shaoul
Date: 2023-02-03

Example script for downloading using the TartanAir dataset toolbox.
"""

# General imports.
import sys

# Local imports.
sys.path.append("..")
import tartanair as ta

# Create a TartanAir object.
tartanair_data_root = "/home/nasciver/tartanair/"

ta.init(tartanair_data_root)

# Download data from following environments.
env = [
    "AbandonedFactory2",
    "AbandonedSchool",
    "AmericanDiner",
    "AmusementPark",
    "AncientTowns",
    "Apocalyptic",
    "BrushifyMoon",
    "CastleFortress",
    "CoalMine",
    "CountryHouse",
    "Cyberpunk",
    "Downtown",
    "Fantasy",
    "ForestEnv",
    "GothicIsland",
    "Hospital",
    "House",
    "HQWesternSaloon",
    "IndustrialHangar",
    "JapaneseCity",
    "ModernCityDowntown",
    "ModularNeighborhood",
    "ModularNeighborhoodIntExt",
    "ModUrbanCity",
    "Nordic",
    "Harbor",
    "Office",
    "OldIndustrialCity",
    "Prison",
    "Restaurant",
    "RetroOffice",
    "Rome",
    "Ruins",
    "SeasideTown",
    "Sewerage",
    "ShoreCaves",
    "Slaughter",
    "TerrainBlending",
    "UrbanConstruction",
    "VictorianStreet",
]
ta.download_multi_thread(
    env=env,
    difficulty=["easy"],
    modality=["image", "depth"],
    camera_name=["lcam_bottom"],
    unzip=True,
    num_workers=8,
)

# Can also download via a yaml config file.
# ta.download(config = 'download_config.yaml')
