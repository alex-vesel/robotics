import os
import re
import numpy as np


def recurse_dir_for_clips(path, match='frames'):
    # get path to last folder containing mp4 files
    clip_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # check if final folder is called 'frames'
            if re.match(match, root.split(os.sep)[-1]):
                # append root up to frames
                clip_paths.append('/'.join(root.split(os.sep)[:-1]))
                break
    
    return list(np.unique(clip_paths))