import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_paths(img_dir, pairs, file_ext='jpg'):
    nrof_skipped_pairs = 0
    path_list = []
    is_same_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(img_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(img_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
            is_same = True
        elif len(pair) == 4:
            path0 = os.path.join(img_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(img_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            is_same = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list.append((path0, path1))
            is_same_list.append(is_same)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        logger.info('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, is_same_list


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines():
            pair = line.strip().split()
            pairs.append(pair)
    return pairs
