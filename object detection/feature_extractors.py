import torchvision
from torchvision.models import ResNet50_Weights,EfficientNet_B2_Weights
def feature_extractor_constructor(feature_extractor_name:str):


    if feature_extractor_name not in ['resnet50', 'efficientnet']:
        return 'feature extractor name not recognized'

    if feature_extractor_name == 'resnet50':
        feature_extractor = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    if feature_extractor_name == 'efficientnet':
        feature_extractor= torchvision.models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)


    return feature_extractor

if __name__=='__main__':
    feature_extractor_constructor('efficientnet')