from model import get_maskrcnn
from PIL import Image, ImageDraw
import numpy as np
from data import PennFudanPed
from torchvision import transforms
import torch

model_path = 'C:\Users\moomo\Documents\AAA Actual Stuff\Programming\Semantic Segmentation Test\models\5-21-2023_20-3\maskrcnn'

model = get_maskrcnn(2)
model.load_state_dict(torch.load(model_path))
model.eval()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)

image_path = 'C:/Users/moomo/Documents/AAA Actual Stuff/Programming/Semantic Segmentation Test/example.png'
image = Image.open(image_path)

predictions = model([(transforms.PILToTensor()(image) / 255).to(device)])

mask_total = np.zeros(predictions[0]['masks'][0, 0].cpu().detach().numpy().shape)
for i in range(len(predictions[0]['masks'])):
    score = predictions[0]['scores'][i]
    if float(score) > 0.9:
        img = predictions[0]['masks'][i, 0].cpu().detach().numpy()
        mask_total += img * 255
img_pred = Image.fromarray(mask_total).convert('RGB')

img_pred.save('pred.png')
img_pred.show()