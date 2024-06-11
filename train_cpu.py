import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import DetrImageProcessor, DetrForObjectDetection

# Custom dataset class for COCO
class COCODataset(Dataset):
    def __init__(self, img_folder, ann_file, transforms):
        self.coco = COCO(ann_file)
        self.img_folder = img_folder
        self.ids = list(self.coco.imgToAnns.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']

        img = Image.open(f'{self.img_folder}/{path}').convert('RGB')
        num_objs = len(anns)

        boxes = []
        labels = []

        for i in range(num_objs):
            xmin = anns[i]['bbox'][0]
            ymin = anns[i]['bbox'][1]
            xmax = xmin + anns[i]['bbox'][2]
            ymax = ymin + anns[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(anns[i]['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["class_labels"] = labels  # Changed key from "labels" to "class_labels"

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

# Transforms
transform = T.Compose([
    T.ToTensor(),
    T.Resize((800, 800)),
])

# Load your dataset
img_folder = 'dataset/images'
ann_file = 'train_annotations.json'
dataset = COCODataset(img_folder, ann_file, transform)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Load the DETR model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, targets in tqdm(data_loader):
        images = [torch.clamp(image, 0, 1) for image in images]  # Clip the image values to [0, 1]
        targets = [{k: v for k, v in t.items()} for t in targets]

        inputs = processor(images, return_tensors="pt", padding=True)
        outputs = model(**inputs, labels=targets)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data_loader)}")

print("Training completed.")