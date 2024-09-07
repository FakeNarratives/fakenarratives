import os
import shutil
import pandas as pd

channel_dir_map = {"BILD TV": "BildTV", "ZDFheute Nachrichten": "HeuteJournal", "WELT": "Welt"}

video2channel = pd.read_csv("video_lists/video2channel.csv")
print(video2channel.head())
## header - , filename, channel

## From location
from_loc = "/nfs/data/fakenarratives/202407_corpus/results_pkl/YouTube"
## To location
to_loc = "/nfs/data/fakenarratives/202407_corpus/results_pkl/"

for i in range(len(video2channel)):
    filename = video2channel.loc[i, "filename"].replace(".mp4", "")
    channel = video2channel.loc[i, "channel"]
    if channel in channel_dir_map:
        channel = channel_dir_map[channel]

    if not os.path.exists(os.path.join(to_loc, channel, filename)) and os.path.exists(os.path.join(from_loc, filename)):
        os.makedirs(os.path.join(to_loc, channel, filename))

    ## Move all files from os.path.join(from_loc, filename) to os.path.join(to_loc, channel, filename)

    if not os.path.exists(os.path.join(from_loc, filename)):
        print(f"{filename} does not exist in {from_loc}")
        continue

    for file in os.listdir(os.path.join(from_loc, filename)):
        shutil.move(os.path.join(from_loc, filename, file), os.path.join(to_loc, channel, filename, file))
        print(f"Moved {file} to {to_loc}/{channel}/{filename}")
