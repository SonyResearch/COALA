# -*-coding:utf-8-*-

import sys
import os
import json

# Common Data format
"""
{
    "filename" :      
                {                  
                    "size" :
                                {
                                    "width" : <string>
                                    "height" : <string>
                                    "depth" : <string>
                                }
                
                    "objects" :
                                {
                                    "num_obj" : <int>
                                    "<index>" :
                                                {
                                                    "name" : <string>
                                                    "bndbox" :
                                                                {
                                                                    "xmin" : <float>
                                                                    "ymin" : <float>
                                                                    "xmax" : <float>
                                                                    "ymax" : <float>
                                                                }
                                                }
                                    ...
                
                
                                }
                }
"""

# XML Data format
"""
{
    "filename" : <XML Object>
    ...
}
"""


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s|%s| %s%% (%s/%s)  %s' % (prefix, bar, percent, iteration, total, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print("\n")


class COCO:
    """
    Handler Class for COCO Format
    """

    @staticmethod
    def parse(json_path):

        try:
            json_data = json.load(open(json_path))

            images_info = json_data["images"]
            cls_info = json_data["categories"]

            data = {}

            progress_length = len(json_data["annotations"])
            progress_cnt = 0
            printProgressBar(0, progress_length, prefix='\nCOCO Parsing:'.ljust(15), suffix='Complete', length=40)

            for anno in json_data["annotations"]:

                image_id = anno["image_id"]
                cls_id = anno["category_id"]

                filename = None
                img_width = None
                img_height = None
                cls = None

                for info in images_info:
                    if info["id"] == image_id:
                        filename, img_width, img_height = \
                            info["file_name"].split(".")[0], info["width"], info["height"]

                for category in cls_info:
                    if category["id"] == cls_id:
                        cls = category["name"]

                size = {
                    "width": img_width,
                    "height": img_height,
                    "depth": "3"
                }

                bndbox = {
                    "xmin": anno["bbox"][0],
                    "ymin": anno["bbox"][1],
                    "xmax": anno["bbox"][2] + anno["bbox"][0],
                    "ymax": anno["bbox"][3] + anno["bbox"][1]
                }

                obj_info = {
                    "name": cls,
                    "bndbox": bndbox
                }

                if filename in data:
                    obj_idx = str(int(data[filename]["objects"]["num_obj"]))
                    data[filename]["objects"][str(obj_idx)] = obj_info
                    data[filename]["objects"]["num_obj"] = int(obj_idx) + 1

                elif filename not in data:

                    obj = {
                        "num_obj": "1",
                        "0": obj_info
                    }

                    data[filename] = {
                        "size": size,
                        "objects": obj
                    }

                printProgressBar(progress_cnt + 1, progress_length, prefix='COCO Parsing:'.ljust(15), suffix='Complete',
                                 length=40)
                progress_cnt += 1

            # print(json.dumps(data, indent=4, sort_keys = True))
            return True, data

        except Exception as e:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            msg = "ERROR : {}, moreInfo : {}\t{}\t{}".format(e, exc_type, fname, exc_tb.tb_lineno)

            return False, msg


class YOLO:
    """
    Handler Class for YOLO Format
    """

    def __init__(self, cls_list_path):
        with open(cls_list_path, 'r') as file:
            l = file.read().splitlines()

        self.cls_list = l

    def coordinateCvt2YOLO(self, size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]

        # (xmin + xmax / 2)
        x = (box[0] + box[1]) / 2.0
        # (ymin + ymax / 2)
        y = (box[2] + box[3]) / 2.0

        # (xmax - xmin) = w
        w = box[1] - box[0]
        # (ymax - ymin) = h
        h = box[3] - box[2]

        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (round(x, 3), round(y, 3), round(w, 3), round(h, 3))

    def parse(self, label_path):
        try:

            (dir_path, dir_names, filenames) = next(os.walk(os.path.abspath(label_path)))

            data = {}

            progress_length = len(filenames)
            progress_cnt = 0
            printProgressBar(0, progress_length, prefix='\nYOLO Parsing:'.ljust(15), suffix='Complete', length=40)

            for filename in filenames:

                txt = open(os.path.join(dir_path, filename), "r")

                filename = filename.split(".")[0]

                img_width = 1280
                img_height = 720
                img_depth = 3

                size = {
                    "width": img_width,
                    "height": img_height,
                    "depth": img_depth
                }

                obj = {}
                obj_cnt = 0

                for line in txt:
                    elements = line.split(" ")
                    name_id = elements[0]

                    xminAddxmax = float(elements[1]) * (2.0 * float(img_width))
                    yminAddymax = float(elements[2]) * (2.0 * float(img_height))

                    w = float(elements[3]) * float(img_width)
                    h = float(elements[4]) * float(img_height)

                    xmin = (xminAddxmax - w) / 2
                    ymin = (yminAddymax - h) / 2
                    xmax = xmin + w
                    ymax = ymin + h

                    bndbox = {
                        "xmin": float(xmin),
                        "ymin": float(ymin),
                        "xmax": float(xmax),
                        "ymax": float(ymax)
                    }

                    obj_info = {
                        "name": name_id,
                        "bndbox": bndbox
                    }

                    obj[str(obj_cnt)] = obj_info
                    obj_cnt += 1

                obj["num_obj"] = obj_cnt

                data[filename] = {
                    "size": size,
                    "objects": obj
                }

                printProgressBar(progress_cnt + 1, progress_length, prefix='YOLO Parsing:'.ljust(15), suffix='Complete',
                                 length=40)
                progress_cnt += 1

            return True, data

        except Exception as e:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            msg = "ERROR : {}, moreInfo : {}\t{}\t{}".format(e, exc_type, fname, exc_tb.tb_lineno)

            return False, msg

    def generate(self, data):

        try:

            progress_length = len(data)
            progress_cnt = 0
            printProgressBar(0, progress_length, prefix='\nYOLO Generating:'.ljust(15), suffix='Complete', length=40)

            result = {}

            for key in data:
                img_width = int(data[key]["size"]["width"])
                img_height = int(data[key]["size"]["height"])

                contents = ""

                for idx in range(0, int(data[key]["objects"]["num_obj"])):
                    xmin = data[key]["objects"][str(idx)]["bndbox"]["xmin"]
                    ymin = data[key]["objects"][str(idx)]["bndbox"]["ymin"]
                    xmax = data[key]["objects"][str(idx)]["bndbox"]["xmax"]
                    ymax = data[key]["objects"][str(idx)]["bndbox"]["ymax"]

                    b = (float(xmin), float(xmax), float(ymin), float(ymax))
                    bb = self.coordinateCvt2YOLO((img_width, img_height), b)

                    cls_id = self.cls_list.index(data[key]["objects"][str(idx)]["name"])

                    bndbox = "".join(["".join([str(e), " "]) for e in bb])
                    contents = "".join([contents, str(cls_id), " ", bndbox[:-1], "\n"])

                result[key] = contents

                printProgressBar(progress_cnt + 1, progress_length, prefix='YOLO Generating:'.ljust(15),
                                 suffix='Complete',
                                 length=40)
                progress_cnt += 1

            return True, result

        except Exception as e:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            msg = "ERROR : {}, moreInfo : {}\t{}\t{}".format(e, exc_type, fname, exc_tb.tb_lineno)

            return False, msg

    def save(self, data, save_path, img_path, img_type, manipast_path):

        try:

            progress_length = len(data)
            progress_cnt = 0
            printProgressBar(0, progress_length, prefix='\nYOLO Saving:'.ljust(15), suffix='Complete', length=40)

            with open(os.path.abspath(os.path.join(manipast_path, "manifast.txt")), "w") as manipast_file:

                for key in data:
                    manipast_file.write(os.path.abspath(os.path.join(img_path, "".join([key, img_type, "\n"]))))

                    with open(os.path.abspath(os.path.join(save_path, "".join([key, ".txt"]))), "w") as output_txt_file:
                        output_txt_file.write(data[key])

                    printProgressBar(progress_cnt + 1, progress_length, prefix='YOLO Saving:'.ljust(15),
                                     suffix='Complete',
                                     length=40)
                    progress_cnt += 1

            return True, None

        except Exception as e:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            msg = "ERROR : {}, moreInfo : {}\t{}\t{}".format(e, exc_type, fname, exc_tb.tb_lineno)

            return False, msg
