from PIL import Image
import numpy as np

img = Image.open('PennFudanPed/PNGImages/FudanPed00001.png')
img.show()

mask = Image.open('PennFudanPed/PedMasks/FudanPed00001_mask.png')
mask = np.array(mask)
mask = np.where(mask > 0, 200, 0)
mask = Image.fromarray(mask)
mask.show()