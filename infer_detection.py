from model import get_fasterrcnn
from PIL import Image, ImageDraw
from torchvision import transforms
import torch

model_path = 'C:/Users/moomo/Documents/AAA Actual Stuff/Programming/Semantic Segmentation Test/models/5-21-2023_11-8/fasterrcnn'

model = get_fasterrcnn(2)
model.load_state_dict(torch.load(model_path))
model.eval()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)

image_path = 'C:/Users/moomo/Documents/AAA Actual Stuff/Programming/Semantic Segmentation Test/example.png'
image = Image.open(image_path)

predictions = model([(transforms.PILToTensor()(image) / 255).to(device)])

img_pred = image.copy()
draw_pred = ImageDraw.Draw(img_pred)
for i in range(len(predictions[0]['boxes'])):
    score = predictions[0]['scores'][i]
    if float(score) > 0.9:
        box = predictions[0]['boxes'][i]
        box = list(box)
        draw_pred.rectangle(box, outline=(255, 0, 0), width=5)
img_pred.save('pred.png')
img_pred.show()