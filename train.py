from model import get_fasterrcnn, get_maskrcnn
from data import PennFudanPed
from tqdm import tqdm
import datetime
import os

import torch
from torchvision import transforms
from torchsummary import summary
from torch.utils.data import DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load model with frozen backbone
model_type = "maskrcnn"
model = get_maskrcnn(2, weights='DEFAULT', freeze_backbone=True)
model.to(device)
## summary(model)

# some parameters
epochs = 50
batch_size = 4
transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.ConvertImageDtype(torch.float)])
collate_fn = lambda batch: tuple(zip(*batch))


data = PennFudanPed('PennFudanPed', transforms=transforms)
train_data, test_data = torch.utils.data.random_split(data, [0.8, 0.2])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# test train forward pass
def test_forward():
    train_loader_test = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    images, targets = next(iter(train_loader_test))
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    model.train()
    output = model(images, targets)
    print(output)

# test test forward pass
def test_inference():
    train_loader_test = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    images, targets = next(iter(train_loader_test))
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    model.eval()
    predictions = model(images)
    print(predictions[0])

# test_forward(); print("----------------------------------------------------------------------------------"); test_inference();

# optimizer
params = [p for p in model.parameters() if p.requires_grad]
optim = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.1)


try:
    os.mkdir('models')
except:
    pass

now = datetime.datetime.now()
try:
    os.mkdir('models/{}-{}-{}_{}-{}'.format(now.month, now.day, now.year, now.hour, now.minute))
except:
    pass


best_loss = 10e10
for epoch in range(epochs):
    # train a single epoch
    print('Epoch {}'.format(epoch+1))
    model.train()
    running_loss = 0
    num_batches = 0
    for images, targets in tqdm(iter(train_loader)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optim.zero_grad()
        losses.backward()
        optim.step()

        running_loss += losses.item()
        num_batches += 1
    train_loss = running_loss / num_batches

    running_loss = 0
    num_batches = 0
    for images, targets in tqdm(iter(test_loader)):
        with torch.no_grad():
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            running_loss += losses.item()
            num_batches += 1
    test_loss = running_loss / num_batches

    print('Training loss: {}, Testing loss: {}'.format(train_loss, test_loss))

    if(test_loss < best_loss):
        torch.save(model.state_dict(), 'models/{}-{}-{}_{}-{}/{}'.format(now.month, now.day, now.year, now.hour, now.minute, model_type))
        best_loss = test_loss