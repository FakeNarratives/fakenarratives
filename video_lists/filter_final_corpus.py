import pandas as pd
import os

old_video_rootpath = "/nfs/data/fakenarratives/202407_corpus/videos"
new_video_rootpath = "/nfs/data/fakenarratives/202409_corpus/videos"

## CompactTV
# comp_df = pd.read_csv('compacttv_tue_thu.csv')
# filt_names = set(comp_df['filename'].tolist())

with open("final_corpus_comp.txt", "r") as fr:
    lines = fr.readlines()
    for line in lines:
        if os.path.exists(os.path.join(new_video_rootpath, "CompactTV", line.strip())):
            continue
        ## Create symbolic link of the video in new_video_rootpath to old_video_rootpath
        old_video_path = os.path.join(old_video_rootpath, "CompactTV", line.strip())
        new_video_path = os.path.join(new_video_rootpath, "CompactTV", line.strip())
        os.symlink(old_video_path, new_video_path)


# other_df = pd.read_csv('youtube_tue_thu.csv')
# filt_names = set(other_df['filename'].tolist())

# ## BildTV
# with open("final_corpus_bild.txt", "w") as fw:
#     for name in ["bild_00", "bild_01"]:
#         with open(f"202407_{name}.txt", "r") as fr:
#             lines = fr.readlines()
#             for line in lines:
#                 if line.strip() in filt_names:
#                     ## Create symbolic link of the video in new_video_rootpath to old_video_rootpath
#                     old_video_path = os.path.join(old_video_rootpath, "BildTV", line.strip())
#                     new_video_path = os.path.join(new_video_rootpath, "BildTV", line.strip())
#                     os.symlink(old_video_path, new_video_path)
#                     fw.write(line)


# ## Welt
# with open("final_corpus_welt.txt", "w") as fw:
#     for name in ["welt_00", "welt_01"]:
#         with open(f"202407_{name}.txt", "r") as fr:
#             lines = fr.readlines()
#             for line in lines:
#                 if line.strip() in filt_names:
#                     ## Create symbolic link of the video in new_video_rootpath to old_video_rootpath
#                     old_video_path = os.path.join(old_video_rootpath, "Welt", line.strip())
#                     new_video_path = os.path.join(new_video_rootpath, "Welt", line.strip())
#                     os.symlink(old_video_path, new_video_path)
#                     fw.write(line)


# ## HeuteJournal
# with open("final_corpus_heute.txt", "w") as fw:
#     for name in ["heute_00", "heute_01"]:
#         with open(f"202407_{name}.txt", "r") as fr:
#             lines = fr.readlines()
#             for line in lines:
#                 if line.strip() in filt_names:
#                     ## Create symbolic link of the video in new_video_rootpath to old_video_rootpath
#                     old_video_path = os.path.join(old_video_rootpath, "HeuteJournal", line.strip())
#                     new_video_path = os.path.join(new_video_rootpath, "HeuteJournal", line.strip())
#                     os.symlink(old_video_path, new_video_path)
#                     fw.write(line)


## Tagesschau
# ## Get list of dates between 20220101 and 20240331 that are either Tuesday or Thursday
# dates = pd.date_range(start="20220101", end="20240331", freq='D')
# dates = dates[dates.weekday.isin([1, 3])]
# ## Get in the format YYYYMMDD
# filt_dates = set([str(date).split(" ")[0].replace("-", "") for date in dates])


# ## Tagesschau videos all
# with open("final_corpus_tag.txt", "w") as fw:
#     for name in ["tag_00", "tag_01", "tag_02", "tag_03"]:
#         with open(f"202407_{name}.txt", "r") as fr:
#             lines = fr.readlines()
#             for line in lines:
#                 vdate = line.strip().split("-")[1] ## Format is YYYY-MM-DD (From 20220101 to 20240331)
#                 ## Check if date is either Tuesday or Thursday
#                 if vdate in filt_dates:
#                     ## Create symbolic link of the video in new_video_rootpath to old_video_rootpath
#                     old_video_path = os.path.join(old_video_rootpath, "Tagesschau", line.strip())
#                     new_video_path = os.path.join(new_video_rootpath, "Tagesschau", line.strip())
#                     os.symlink(old_video_path, new_video_path)
#                     fw.write(line)

