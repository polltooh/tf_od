from __future__ import division

import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import xmltodict
import json


DEBUG = False


def meta_property():
    classes = ['vehicle']
    meta = {
        'dataset_name': 'WebCamT',
        'image_width': 352,
        'image_height': 240,
        'classes': classes,
    }
    return meta


def draw_boxes(image, boxes, c, thickness):
    box_num = boxes.shape[0]
    for i in range(box_num):
        cv2.rectangle(image, (int(boxes[i][0]), int(boxes[i][1])),
                      (int(boxes[i][2]), int(boxes[i][3])), c, thickness)
    return image


def order_dict_2_box(order_dict):
    bbox = order_dict['bndbox']
    # box = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
    box = [bbox['ymin'], bbox['xmin'], bbox['ymax'], bbox['xmax']]
    box = [float(b) for b in box]
    return box


def parse_annot(annot_name, max_num_box):
    assert(annot_name.endswith(".xml"))
    with open(annot_name) as xml_d:
        ss = xml_d.read()
        try:
            doc = xmltodict.parse(ss)
        except:
            try:
                ss = ss.replace("&", "")
                doc = xmltodict.parse(ss)
            except:
                print(annot_name + " cannot be read")

    bbox_list = list()
    type_list = list()
    if 'vehicle' not in doc['annotation']:
        print(annot_name + " no vehicle")
    else:
        if isinstance(doc['annotation']['vehicle'], list):
            for vehicle in doc['annotation']['vehicle']:
                box = order_dict_2_box(vehicle)
                vehicle_type = int(vehicle['type'])
                bbox_list.append(box)
                type_list.append(vehicle_type)
        else:
            vehicle = doc['annotation']['vehicle']
            vehicle_type = int(vehicle['type'])
            bbox_list = [order_dict_2_box(vehicle)]
            type_list.append(vehicle_type)

    max_num_box[0] = max(max_num_box[0], len(bbox_list))
    image_name = annot_name.replace('.xml', '.jpg')
    mask_name = '/'.join(image_name.split('/')[:-1]) + '_mask.png'

    file_dict = {}
    file_dict['labels'] = type_list
    file_dict['bboxes'] = bbox_list
    file_dict['image_name'] = image_name.encode('utf-8')
    file_dict['mask_name'] = mask_name.encode('utf-8')

    # image = cv2.imread(image_name)
    # if image is None:
    #     print("no image", image_name)
    #     file_dict = {}
    # else:
    #     if image.shape != (240, 352, 3):
    #         print(image_name, image.shape)
    #         image = cv2.resize(image, (352, 240))
    #         cv2.imwrite(image_name, image)

    #         if DEBUG:
    #             draw_boxes(image, box_array, (255, 0, 0), 2)
    #             cv2.imshow("img", image)
    #             cv2.waitKey(0)

    return file_dict


def full_path_listdir(data_dir):
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir)]


def txt_to_json(data_dir, max_num_box):
    print('Processing {}'.format(data_dir))
    json_list = []
    cam_list = [f for f in full_path_listdir(data_dir) if os.path.isdir(f)]
    for cam in cam_list:
        full_path = cam
        annotation_list = [os.path.join(full_path, f) for f in os.listdir(
            full_path) if f.endswith('xml')]
        for annot in annotation_list:
            file_dict = parse_annot(annot, max_num_box)
            if file_dict != {}:
                json_list.append(file_dict)

    return json_list


if __name__ == "__main__":
    meta = meta_property()
    meta_json = {'meta': meta}
    max_num_box = [0]
    data_dir = '../data/'
    park_train_json = txt_to_json(os.path.join(data_dir, '164'), max_num_box)

    meta_json['meta']['max_num_box'] = max_num_box[0]
    meta_json['meta']['num_classes'] = max(
        [max(anno['labels']) for anno in park_train_json])

    with open('../file_list/164.json', 'w') as f:
        json.dump(park_train_json, f, indent=4)

    # with open('../file_list/WebCamT_Parkway_Test.json', 'w') as f:
    #     json.dump(park_test_json, f, indent=4)

    # with open('../file_list/WebCamT_Downtown_Train.json', 'w') as f:
    #     json.dump(downtown_train_json, f, indent=4)

    # with open('../file_list/WebCamT_Downtown_Test.json', 'w') as f:
    #     json.dump(downtown_test_json, f, indent=4)

    with open('../file_list/WebCamT_meta.json', 'w') as f:
        json.dump(meta_json, f, indent=4)
