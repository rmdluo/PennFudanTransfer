from model import get_fasterrcnn
from PIL import Image, ImageDraw
from data import PennFudanPed
from torchvision import transforms
import torch

path = 'C:/Users/moomo/Documents/AAA Actual Stuff/Programming/Semantic Segmentation Test/models/5-21-2023_11-8/fasterrcnn'

model = get_fasterrcnn(2)
model.load_state_dict(torch.load(path))
model.eval()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)

data = PennFudanPed('PennFudanPed', None)
image, targets = data[0]

predictions = model([transforms.PILToTensor()(image) / 255])

# print(predictions)

img_truth = image.copy()
img_pred = image.copy()

draw_truth = ImageDraw.Draw(img_truth)
for box in targets['boxes']:
    box = list(box)
    draw_truth.rectangle(box, outline=(255, 0, 0))
img_truth.save('truth.png')

draw_pred = ImageDraw.Draw(img_pred)
for box in predictions[0]['boxes']:
    box = list(box)
    draw_pred.rectangle(box, outline=(255, 0, 0))
img_pred.save('pred.png')