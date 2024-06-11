import os
import random
import json

def split_data(images_dir, annotations_file, train_ratio=0.8):
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    # Create train and test directories
    train_dir = 'train'
    test_dir = 'test'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Shuffle data
    random.shuffle(images)

    # Split data
    num_train = int(len(images) * train_ratio)
    train_images = images[:num_train]
    test_images = images[num_train:]

    # Move images to respective directories
    for image in train_images:
        filename = image['file_name']
        os.rename(os.path.join(images_dir, filename), os.path.join(train_dir, filename))

    for image in test_images:
        filename = image['file_name']
        os.rename(os.path.join(images_dir, filename), os.path.join(test_dir, filename))

    # Update image paths in annotations
    for image in train_images:
        image['file_name'] = os.path.join(train_dir, image['file_name'])
    for image in test_images:
        image['file_name'] = os.path.join(test_dir, image['file_name'])

    # Update annotation file
    train_annotations = [annotation for annotation in annotations if annotation['image_id'] in [img['id'] for img in train_images]]
    test_annotations = [annotation for annotation in annotations if annotation['image_id'] in [img['id'] for img in test_images]]

    train_data = {'images': train_images, 'annotations': train_annotations, 'categories': categories}
    test_data = {'images': test_images, 'annotations': test_annotations, 'categories': categories}

    with open('train_annotations.json', 'w') as f:
        json.dump(train_data, f)

    with open('test_annotations.json', 'w') as f:
        json.dump(test_data, f)

if __name__ == "__main__":
    split_data('dataset/images', 'coco_annotations.json')