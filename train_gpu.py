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
    def __init__(self, img_folder, ann_file, transforms=None):
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
        coco_annotations = []

        for i in range(num_objs):
            xmin = anns[i]['bbox'][0]
            ymin = anns[i]['bbox'][1]
            xmax = xmin + anns[i]['bbox'][2]
            ymax = ymin + anns[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(anns[i]['category_id'])
            area = anns[i]['bbox'][2] * anns[i]['bbox'][3]
            coco_annotations.append({
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                'category_id': anns[i]['category_id'],
                'area': area,
                'iscrowd': 0,
                'image_id': img_id,
                'id': anns[i]['id']
            })

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'image_id': torch.tensor([img_id]),
            'annotations': coco_annotations
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

# Transforms
transform = T.Compose([
    T.ToTensor(),
    T.Resize((800, 800))
])

# Load your dataset
img_folder = 'E:/DETR/train'
ann_file = 'train_annotations.json'
dataset = COCODataset(img_folder, ann_file, transform)

# Collate function
def collate_fn(batch):
    images, targets = zip(*batch)
    images = [image.to('cuda') for image in images]
    targets = [{k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
    return images, targets

data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Load the DETR model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").cuda()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, targets in tqdm(data_loader):
        # Move images and annotations to CPU for processing
        images = [image.cpu() for image in images]
        targets = [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # Clip image values to the range [0, 1]
        images = [image.clamp(0, 1) for image in images]

        # Prepare the images and annotations for the model
        inputs = processor(images=images, annotations=targets, return_tensors="pt", padding=True)

        # Move all tensors in the inputs dictionary to the GPU
        for k, v in inputs.items():
            if isinstance(v, list):
                inputs[k] = [item.to('cuda') for item in v]
            else:
                inputs[k] = v.to('cuda')

        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data_loader)}")

# Save the final model
torch.save(model.state_dict(), 'detr_final_model.pth')

print("Training completed and final model saved.")