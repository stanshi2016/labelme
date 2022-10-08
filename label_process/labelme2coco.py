
# @Description: labelme dataset to coco, reference from https://github.com/wkentaro/labelme/tree/main/examples/instance_segmentation
# @Author     : stan.shi
# @Time       : 2022/10/01 08:00 上午

#!/usr/bin/env python
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid

import imgviz
import numpy as np
from sklearn.model_selection import train_test_split
import labelme
import copy

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)

# 0为背景
classname_to_id = {
    "lanqi_s001": 1,
}

coco_instance_data_format = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=None,
            contributor=None,
            date_created=None,
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

def to_coco(input_dir, json_files, output_dir, train_instance=True ,noviz=False):
    now = datetime.datetime.now()
    data = copy.deepcopy(coco_instance_data_format)
    data["info"]["year"] = now.year
    data["info"]["date_created"] = now.strftime("%Y-%m-%d %H:%M:%S.%f")

    #categories
    for k, v in classname_to_id.items():
        data["categories"].append(
            dict(supercategory=None, id=v, name=k,)
        )
    
    if train_instance:
        out_annotation_file = os.path.join(output_dir, "annotations", "instances_train2017.json")
    else:
        out_annotation_file = os.path.join(output_dir, "annotations", "instances_val2017.json")

    #label_files = glob.glob(osp.join(input_dir, "*.json"))
    for image_id, filename in enumerate(json_files):
        print("Generating dataset from:", filename)
        label_file = labelme.LabelFile(filename=filename)
        base = osp.splitext(osp.basename(filename))[0]
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        if train_instance:
            imgviz.io.imsave("{}/images/train2017/{}".format(output_dir, base + ".jpg"),img)
        else:
            imgviz.io.imsave("{}/images/val2017/{}".format(output_dir, base + ".jpg"),img)
        
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name= base + ".jpg",
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
            if shape_type == "circle":
                (x1, y1), (x2, y2) = points
                r = np.linalg.norm([x2 - x1, y2 - y1])
                # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
                # x: tolerance of the gap between the arc and the line segment
                n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
                i = np.arange(n_points_circle)
                x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
                y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
                points = np.stack((x, y), axis=1).flatten().tolist()
            else:
                points = np.asarray(points).flatten().tolist()

            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in classname_to_id:
                continue
            cls_id = classname_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )

        if not noviz:
            viz = img
            if masks:
                labels, captions, masks = zip(
                    *[
                        (classname_to_id[cnm], cnm, msk)
                        for (cnm, gid), msk in masks.items()
                        if cnm in classname_to_id
                    ]
                )
                viz = imgviz.instances2rgb(
                    image=img,
                    labels=labels,
                    masks=masks,
                    captions=captions,
                    font_size=15,
                    line_width=2,
                )
            out_viz_file = osp.join(
                output_dir, "Visualization", base + ".jpg"
            )
            imgviz.io.imsave(out_viz_file, viz)
    with open(out_annotation_file, "w",encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)  # indent=2 更加美观显示

def main():
    input_dir = './data_annotated'
    output_dir = "./data_coco3"
    # 创建文件
    for item in ['images']:
        for subset in ['train2017', 'val2017']:
            os.makedirs(os.path.join(output_dir, item, subset),exist_ok=True)

    for item in ['annotations']:
            os.makedirs(os.path.join(output_dir, item),exist_ok=True)
   
    # 获取images目录下所有的json文件列表
    print(input_dir + "/*.json")
    json_list_path = glob.glob(input_dir + "/*.json")
    print('json_list_path: ', len(json_list_path))
    if (len(json_list_path) <= 0):
        return

    # 数据划分,这里没有区分val2017和tran2017目录，所有图片都放在images目录下
    train_files, val_files = train_test_split(json_list_path, test_size=0.1, train_size=0.9)
    print("train_n:", len(train_files), 'val_n:', len(val_files))
    
    # to colo dataset
    #   dataset:
    #   |--- annotations
    #       |--- instances_train2017.json
    #       |--- instances_val2017.json
    #   |--- images
    #       |--- train2017
    #       |--- val2017
    to_coco(input_dir, train_files, output_dir, train_instance=True, noviz=False)
    to_coco(input_dir, val_files, output_dir, train_instance=False, noviz=False)

if __name__ == "__main__":
    main()
