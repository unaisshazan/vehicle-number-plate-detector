import os
from pathlib import Path
from shutil import copyfile

main_dir = "dataset/UFPR-ALPR"
target_dir = "dataset/UFPR-ALPR-snapshots"
photos_indices = ["[10]", "[20]"]

for tracks_directory_name in os.listdir(main_dir):
    if not tracks_directory_name.startswith("."):
        for file_name in os.listdir("{}/{}".format(main_dir, tracks_directory_name)):
            if file_name.endswith(".png"):
                if any(index_string in file_name for index_string in photos_indices):
                    print("{}/{}".format(target_dir, file_name))
                    Path("{}/{}".format(target_dir, file_name)).touch()
                    copyfile(
                        src="{}/{}/{}".format(main_dir, tracks_directory_name, file_name),
                        dst="{}/{}".format(target_dir, file_name)
                    )