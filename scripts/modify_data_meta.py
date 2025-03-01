import json
import orjson
from pathlib import Path
import os
import sys
sys.path.append(".")

from shared_utils.path_utils import recurse_dir_for_clips
from behavior_cloning.configs.path_config import DATA_DIR


data_path = Path(DATA_DIR)
clip_paths = recurse_dir_for_clips(data_path, match='rgb_frames')

TASK_DESCRIPTIONS = [
    "Pick up the earplug and return home.",
    "Pick up the orange earplug and return home.",
    "Pick up the earplug and return to your starting position.",
    "Pick up the orange earplug and return to your starting position.",
    "Pick up the earplug and return to the starting position.",
    "Pick up the orange earplug and return to the starting position.",
    "Pick up the earplug and return to the starting position.",
    "Grab the earplug and head back home.",
    "Pick up the orange earplug and make your way home.",
    "Take the earplug and return to your starting location.",
    "Pick up the orange earplug and go back to your starting point.",
    "Grab the earplug and head back to your starting position.",
    "Pick up the orange earplug and return to the starting position.",
    "Pick up the earplug and return to where you started.",
    "Pick up the orange earplug and head back to your original spot.",
]

TASK_NAME = "earplug_return_home"

for clip_path in clip_paths:
    meta_path = os.path.join(clip_path, 'meta')
    for meta_file in os.listdir(meta_path):
        with open(os.path.join(meta_path, meta_file)) as f:
            meta = orjson.loads(f.read())

        # if meta['gripper_has_object'] is not None:
        #     meta['task_description'] = None
        # else:
        #     meta['task_description'] = TASK_DESCRIPTIONS

        if meta['gripper_has_object'] is not None:
            meta['task_name'] = "object_classification"
        else:
            meta['task_name'] = TASK_NAME

        with open(os.path.join(meta_path, meta_file), 'w') as f:
            json.dump(meta, f)