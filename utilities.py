import os
import json
from PIL import Image


def get_image_info(file_name, image_id, img_folder):
    image = Image.open(os.path.join(img_folder, file_name))
    width, height = image.size
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height
    }


def get_category_info(class_id, categories):
    category = {
        "id": class_id,
        "name": str(class_id)
    }
    if category not in categories:
        categories.append(category)


def get_annotation_info(ann_id, image_id, class_id, bbox, width, height):
    x_center, y_center, w, h = bbox
    x_min = (x_center - w / 2) * width
    y_min = (y_center - h / 2) * height
    bbox = [x_min, y_min, w * width, h * height]
    area = bbox[2] * bbox[3]
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": class_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0
    }


def yolo_to_coco(yolo_folder, img_folder, output_file):
    images = []
    annotations = []
    categories = []

    ann_id = 1
    image_id = 1

    for filename in os.listdir(yolo_folder):
        if not filename.endswith('.txt'):
            continue

        img_filename = filename.replace('.txt', '.png')
        image_info = get_image_info(img_filename, image_id, img_folder)
        images.append(image_info)

        with open(os.path.join(yolo_folder, filename)) as file:
            for line in file:
                parts = line.strip().split()
                class_id = int(parts[0])
                bbox = list(map(float, parts[1:]))

                annotation = get_annotation_info(ann_id, image_id, class_id, bbox, image_info['width'],
                                                 image_info['height'])
                annotations.append(annotation)
                get_category_info(class_id, categories)

                ann_id += 1

        image_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_file, 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)


# Define your paths
yolo_folder = 'dataset/labels'
img_folder = 'dataset/images'
output_file = 'coco_annotations_batch.json'
# Convert YOLO to COCO
yolo_to_coco(yolo_folder, img_folder, output_file)