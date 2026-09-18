# each da Vinci image has a border of dark pixels on each side, and a menu on the bottom displaying some information.
# as part of the preprocessing, we will crop the center of the image to remove the borders and menu.
# all the images have the same shape of 3x1920x1080, and will be cropped to 3x1280x944, with the center of the image being preserved. Bottom is 
# cropped more due to the menu, and the top is cropped less to preserve the surgical field of view. 

import cv2
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm


def center_crop_and_save(image_path, new_width=1340, new_height=1080, bottom_crop=140):
    img = cv2.imread(image_path)
    if img is None:
        return
    height, width = img.shape[:2]
    if width != 1920 or height != 1080:
        return
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height - bottom_crop) // 2
    cropped = img[top:bottom, left:right]
    cv2.imwrite(image_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])

samples_path = [
    "/.../surgical_video_assessment/turbo_data/tension_clips",
    "/.../surgical_video_assessment/turbo_data/non_tensions_clips"
]

all_paths = []
for sample_path in samples_path:
    for folder in sorted(os.listdir(sample_path)):
        folder_path = os.path.join(sample_path, folder, "frames")
        if os.path.isdir(folder_path):
            for file in sorted(os.listdir(folder_path)):
                if file.endswith(".jpg") or file.endswith(".png"):
                    all_paths.append(os.path.join(folder_path, file))

print(f"Processing {len(all_paths)} images with {cpu_count()} workers")

with Pool(cpu_count()) as pool:
    list(tqdm.tqdm(pool.imap_unordered(center_crop_and_save, all_paths, chunksize=64), total=len(all_paths))) 
                