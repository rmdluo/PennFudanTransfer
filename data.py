import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

class PennFudanPed(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.img_paths = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.annotation_paths = list(sorted(os.listdir(os.path.join(root, 'Annotation'))))
        self.mask_paths = list(sorted(os.listdir(os.path.join(root, 'PedMasks'))))
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'PNGImages', self.img_paths[idx])
        mask_path = os.path.join(self.root, 'PedMasks', self.mask_paths[idx])

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.transforms:
            img = self.transforms(img)
            mask = self.transforms(mask)

        mask = np.array(mask)
        if(len(mask.shape) > 2):
            mask = np.squeeze(mask)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        boxes = []
        masks = []
        for obj in obj_ids:
            obj_mask = np.where(mask == obj, True, False)
            masks.append(obj_mask)
            pos = np.nonzero(obj_mask)
            if(len(obj_mask.shape) > 2):
                xmin = np.min(pos[2])
                xmax = np.max(pos[2])
                ymin = np.min(pos[1])
                ymax = np.max(pos[1])
            else:
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones(len(obj_ids), dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros(len(obj_ids), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        return img, target

if __name__=='__main__':
    data = PennFudanPed('PennFudanPed', None)
    _, target = data[0]
    print(target['masks'].size())
    # for i in range(data.__len__()):
    #     data.__getitem__(i)
