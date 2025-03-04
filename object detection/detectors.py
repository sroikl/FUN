import torch
from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
                                          maskrcnn_resnet50_fpn_v2,MaskRCNN_ResNet50_FPN_V2_Weights,)

def detector_constructor(detector_name:str):

    """ construct a detector and give back the class names"""
    if detector_name not in ['fasterrcnn', 'maskrcnn','yolo']:
        return 'model name not recognized'
    
    if detector_name == 'fasterrcnn':
        detector = fasterrcnn_resnet50_fpn_v2(weights= FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        class_names= FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]
    if detector_name == 'maskrcnn':
        detector= maskrcnn_resnet50_fpn_v2(weights= MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        class_names= MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]
    if detector_name == 'yolo':
        detector= torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        class_names= MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"] #yolo was traind on COCO

    return detector, class_names




if __name__ == '__main__':
    detector, class_names = detector_constructor('yolo')