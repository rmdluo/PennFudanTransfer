import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_fasterrcnn(num_classes, weights=None, freeze_backbone=True):
    # load model and freeze backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # model.backbone.requires_grad = False -- THIS DOESN'T WORK!
    for param in model.backbone.parameters():
        param.requires_grad = False

    return model