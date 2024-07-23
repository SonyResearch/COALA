"""
These codes are adopted from LEAF with some modifications.

It converts a list of (writer, [list of (file,class)]) tuples into a json object of the form:
  {users: [bob, etc], num_samples: [124, etc.],
  user_data: {bob : {x:[img1,img2,etc], y:[class1,class2,etc]}, etc}},
where "img_" is a vectorized representation of the corresponding image.
"""

from __future__ import division

import json
import math
import os

import numpy as np
from PIL import Image

from coala.datasets.utils import util

MAX_WRITERS = 100  # max number of writers per json file.


def relabel_class(c):
    """
    maps hexadecimal class value (string) to a decimal number
    returns:
    - 0 through 9 for classes representing respective numbers
    - 10 through 35 for classes representing respective uppercase letters
    - 36 through 61 for classes representing respective lowercase letters
    """
    if c.isdigit() and int(c) < 40:
        return int(c) - 30
    elif int(c, 16) <= 90:  # uppercase
        return int(c, 16) - 55
    else:
        return int(c, 16) - 61


def data_to_json(base_folder):
    by_writer_dir = os.path.join(base_folder, "intermediate", "images_by_writer")
    writers = util.load_obj(by_writer_dir)

    num_json = int(math.ceil(len(writers) / MAX_WRITERS))

    users = []
    num_samples = []
    user_data = {}

    writer_count = 0
    json_index = 0
    for (w, l) in writers:

        users.append(w)
        num_samples.append(len(l))
        user_data[w] = {"x": [], "y": []}

        size = 28, 28  # original image size is 128, 128
        for (f, c) in l:
            file_path = os.path.join(base_folder, f)
            img = Image.open(file_path)
            gray = img.convert("L")
            gray.thumbnail(size, Image.ANTIALIAS)
            arr = np.asarray(gray).copy()
            vec = arr.flatten()
            vec = vec / 255  # scale all pixel values to between 0 and 1
            vec = vec.tolist()

            nc = relabel_class(c)

            user_data[w]["x"].append(vec)
            user_data[w]["y"].append(nc)

        writer_count += 1
        if writer_count == MAX_WRITERS:
            all_data = {}
            all_data["users"] = users
            all_data["num_samples"] = num_samples
            all_data["user_data"] = user_data

            file_name = "all_data_%d.json" % json_index
            file_path = os.path.join(base_folder, "all_data", file_name)

            print("writing %s" % file_name)

            with open(file_path, "w") as outfile:
                json.dump(all_data, outfile)

            writer_count = 0
            json_index += 1

            users[:] = []
            num_samples[:] = []
            user_data.clear()
