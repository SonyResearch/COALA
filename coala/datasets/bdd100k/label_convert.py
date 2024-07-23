import os
import json
from tqdm import tqdm
from coala.datasets.bdd100k.format import COCO, YOLO


def bdd2coco_detection(id_dict, labeled_images, fn, attr_dict):
    images = list()
    annotations = list()

    counter = 0
    for i in tqdm(labeled_images):
        counter += 1
        image = dict()
        image['file_name'] = i['name']
        image['height'] = 720
        image['width'] = 1280

        image['id'] = counter

        empty_image = True

        for label in i['labels']:
            annotation = dict()
            category = label['category']
            if (category == "traffic light"):
                color = label['attributes']['trafficLightColor']
                category = "tl_" + color
            if category in id_dict.keys():
                empty_image = False
                annotation["iscrowd"] = 0
                annotation["image_id"] = image['id']
                x1 = label['box2d']['x1']
                y1 = label['box2d']['y1']
                x2 = label['box2d']['x2']
                y2 = label['box2d']['y2']
                annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                annotation['area'] = float((x2 - x1) * (y2 - y1))
                annotation['category_id'] = id_dict[category]
                annotation['ignore'] = 0
                annotation['id'] = label['id']
                annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                annotations.append(annotation)

        if empty_image:
            continue

        images.append(image)

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    print('saving...')
    json_string = json.dumps(attr_dict)
    with open(fn, "w") as file:
        file.write(json_string)


def bdd_to_coco_convert(label_dir, save_path):
    # label_dir = "data/bdd100k/labels/"
    # save_path = "data/bdd100k/coco_labels/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    attr_dict = dict()
    attr_dict["categories"] = [
        {"supercategory": "none", "id": 1, "name": "person"},
        {"supercategory": "none", "id": 2, "name": "rider"},
        {"supercategory": "none", "id": 3, "name": "car"},
        {"supercategory": "none", "id": 4, "name": "bus"},
        {"supercategory": "none", "id": 5, "name": "truck"},
        {"supercategory": "none", "id": 6, "name": "bike"},
        {"supercategory": "none", "id": 7, "name": "motor"},
        {"supercategory": "none", "id": 8, "name": "tl_green"},
        {"supercategory": "none", "id": 9, "name": "tl_red"},
        {"supercategory": "none", "id": 10, "name": "tl_yellow"},
        {"supercategory": "none", "id": 11, "name": "tl_none"},
        {"supercategory": "none", "id": 12, "name": "traffic sign"},
        {"supercategory": "none", "id": 13, "name": "train"}
    ]

    attr_id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

    # create BDD training set detections in COCO format
    print('Loading training set...')
    with open(os.path.join(label_dir,
                           'bdd100k_labels_images_train.json')) as f:
        train_labels = json.load(f)
    print('Converting training set...')

    out_fn = os.path.join(save_path,
                          'bdd100k_labels_images_det_coco_train.json')
    bdd2coco_detection(attr_id_dict, train_labels, out_fn, attr_dict)

    print('Loading validation set...')
    # create BDD validation set detections in COCO format
    with open(os.path.join(label_dir,
                           'bdd100k_labels_images_val.json')) as f:
        val_labels = json.load(f)
    print('Converting validation set...')

    out_fn = os.path.join(save_path,
                          'bdd100k_labels_images_det_coco_val.json')
    bdd2coco_detection(attr_id_dict, val_labels, out_fn, attr_dict)


def coco_to_yolo_convert(output_dir, coco_label_dir):
    # output_paths = {
    #     "train": "data/bdd100k/labels/train/",
    #     "val": "data/bdd100k/labels/val/"
    # }

    root_path = os.path.dirname(os.path.abspath(__file__))
    name_path = os.path.join(root_path, "bdd100k.names")

    for dataset in ["train", "val"]:
        config = {
            "datasets": "COCO",
            "img_path": "data/bdd100k/images/100k/{}".format(dataset),
            "label": os.path.join(coco_label_dir, "bdd100k_labels_images_det_coco_{}.json".format(dataset)),
            "img_type": ".jpg",
            "manipast_path": "./",
            "output_path": os.path.join(output_dir, dataset),
            "cls_list": name_path,
        }

        if not os.path.exists(config["output_path"]):
            os.makedirs(config["output_path"])

        if config["datasets"] == "COCO":
            coco = COCO()
            yolo = YOLO(os.path.abspath(config["cls_list"]))

            flag, data = coco.parse(config["label"])
            print("parsed")

            if flag:
                flag, data = yolo.generate(data)
                print("generated")

                if flag:
                    flag, data = yolo.save(data, config["output_path"], config["img_path"],
                                           config["img_type"], config["manipast_path"])
                    print("saved")

                    if not flag:
                        print("Saving Result : {}, msg : {}".format(flag, data))

                else:
                    print("YOLO Generating Result : {}, msg : {}".format(flag, data))

            else:
                print("COCO Parsing Result : {}, msg : {}".format(flag, data))

        else:
            print("Unknown Datasets")
