import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
import torchvision.transforms as T

# Load the trained model
model_path = 'detr_model.pth'
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.load_state_dict(torch.load(model_path))
model.eval()

# Load the image processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]  # Save original size
    image = cv2.resize(image, (800, 800))
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image, original_size

# Function to plot the image with bounding boxes
def plot_results(image, boxes, labels, scores):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # You can adjust the threshold as needed
            xmin, ymin, xmax, ymax = box
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin, ymin, f'{label} ({score:.2f})', color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.axis('off')
    plt.show()

# Test image path
test_image_path = 'test/0016_0321123728_02_WRI-L1_F010.png'

# Preprocess the image
image, original_size = preprocess_image(test_image_path)
image = image.unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    outputs = model(pixel_values=image)

# Process the model outputs
probabilities = outputs.logits.softmax(-1)[0, :, :-1]
boxes = outputs.pred_boxes[0]

# Filter out predictions with low confidence
threshold = 0.5
labels = torch.argmax(probabilities, dim=-1)
scores = torch.max(probabilities, dim=-1).values
keep = scores > threshold
boxes = boxes[keep]
labels = labels[keep]
scores = scores[keep]

# Convert boxes from [0, 1] to original image scales
boxes = boxes.cpu().numpy()
boxes[:, 0] *= original_size[1]  # xmin
boxes[:, 1] *= original_size[0]  # ymin
boxes[:, 2] *= original_size[1]  # xmax
boxes[:, 3] *= original_size[0]  # ymax

# Load the original image for plotting
original_image = cv2.imread(test_image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Plot the results
plot_results(original_image, boxes, labels, scores)