from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
import numpy as np
import hashlib
import os
import json
from tqdm import tqdm
sys.path.append(".")

from shared_utils.path_utils import recurse_dir_for_clips
from behavior_cloning.configs.path_config import DATA_DIR, TASK_DESCRIPTION_CACHE_PATH


data_path = Path(DATA_DIR)
clip_paths = recurse_dir_for_clips(data_path, match='rgb_frames')

# get all unique task descriptions
task_description_set = set()
print("Getting all unique task descriptions")
for clip_path in tqdm(clip_paths):
    meta_path = os.path.join(clip_path, 'meta')

    for meta_file in os.listdir(meta_path):
        with open(os.path.join(meta_path, meta_file)) as f:
            meta = json.load(f)

        if 'task_description' in meta and meta['task_description'] is not None:
            for task_description in meta['task_description']:
                task_description_set.add(task_description)

# encode descriptions
os.makedirs(TASK_DESCRIPTION_CACHE_PATH, exist_ok=True)

print("Encoding task descriptions")
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

for task_description in tqdm(task_description_set):
    encoded_task_description = model.encode(task_description).astype(np.float32, copy=False)
    description_hash = hashlib.md5(task_description.encode()).hexdigest()
    with open(os.path.join(TASK_DESCRIPTION_CACHE_PATH, f'{description_hash}.npy'), 'wb') as f:
        np.save(f, encoded_task_description)
