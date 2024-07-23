import glob
from pathlib import Path
import os


def load_files_paths(path):
    try:
        f = []  # image files
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            elif p.is_file():  # file
                with open(p, 'r') as t:
                    t = t.read().strip().splitlines()
                    f += [x for x in t]
            else:
                raise FileNotFoundError(f'{p} does not exist')
        im_files = sorted(x.replace('/', os.sep) for x in f)
        assert im_files, f'No files found'
    except Exception as e:
        raise Exception(f'Error loading data from {path}')

    return im_files