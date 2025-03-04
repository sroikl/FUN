import torch
from torch import nn

from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

class ObjectEmbedder:
    """
    this class is an implementation of an object embedder in 2 steps:
        1.detect all the objects in the image with an object detector (specifically faster-rcnn)
        2. embed the relevant objects in the image with a feature extractor (specifically ResNet50)
    """

    def __init__(
            self,
            detector: nn.Module,
            feature_extractor: nn.Module,
            detection_th: float = 0.5,
            device: Optional[str] = None,
            embedding_dim: int = 2048,
            fe_input_dim:tuple = (224, 224),#typical for resnet
            class_names: Optional[List[str]] = None,
            detetor_name: str = 'fasterrcnn',
    ):
        """

        :param detector: an nn object detector
        :param feature_extractor: an nn object feature extractor
        :param detection_th: the minimum detection probability threshold
        :param device: a string either "cpu" or "cuda"
        :param embedding_dim: int. dimension of embedding vector
        :param fe_input_dim: int. dimension of feature extractor input
        :param class_names: list of strings. names of classes
        """

        # Set the device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Set detection threshold
        self.detection_threshold = detection_th

        # Initialize the object detector (Faster R-CNN)
        self.detector = detector
        self.detector.eval()
        self.detector.to(self.device)

        # Remove the classification head
        self.feature_extractor = feature_extractor
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)


        self.embedding_dim = embedding_dim
        self.class_names = class_names
        self.fe_input_dim = fe_input_dim
        self.detector_name = detetor_name
    def _preprocess_im(self, image: Image.Image) -> torch.Tensor:
        """ pil->torch.tensor """

        img_tensor = F.to_tensor(image)
        return img_tensor.to(self.device)

    def _extract_features_from_roi(self, image: Image.Image,boxes:torch.Tensor) -> torch.Tensor:
        """ for each region in the image i.e. image[bbox] we encode it using our self.feature_extractor to get features"""

        #collect all the regions
        regions= []
        for box in boxes:
            x1,y1,x2,y2 = box.int().tolist()
            region= image.crop((x1,y1,x2,y2))
            region= region.resize(self.fe_input_dim)
            regions.append(F.to_tensor(region))

        if not regions: #no regions detected
            return torch.zeros((0, self.embedding_dim), device=self.device)

        region_batch = torch.stack(regions).to(self.device)
        with torch.no_grad():
            features= self.feature_extractor(region_batch) #[B,C,1,1]->[B,C]
            features= features.squeeze(-1).squeeze(-1)

        return features

    def _normalize_roi_boxes(self, boxes: torch.Tensor,image_size: Tuple[int, int]) -> torch.Tensor:
        """normalize the roi boxes based on image measurments"""

        width, height = image_size
        norm_boxes= boxes.clone()
        norm_boxes[:, 0] /= width
        norm_boxes[:, 1] /= height
        norm_boxes[:, 2] /= width
        norm_boxes[:, 3] /= height

        return norm_boxes

    def _filter_detections(self,boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ filter the detections based on confidence score and self.detection_threshold"""

        pass_idx= scores > self.detection_threshold
        return boxes[pass_idx], scores[pass_idx], labels[pass_idx]

    def _get_class_names(self,class_ids:torch.Tensor) -> List[str]:
        """class ids -> class names for visualization"""

        return [self.class_names[int(idx)] for idx in class_ids]

    def __call__(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:

        """ detect ROIs in input image and returns the bbox as well as the embedding of each roi
        output:
            - torch.Tensor of shape (k, 6) where k is the number of objects detected
            6 is the number of attributes of each object: [x1, y1, x2, y2, confidence, class]
            xyxy is normalized. (0, 0) is top-left and (1, 1) is bottom-right
            - torch.Tensor of shape (k,n) where k is the number of objects detected
            and n is the dimension of your embedding space
        """

        width, height = image.size

        if self.detector_name == 'yolo':
            detections = self.detector([image]).xyxy[0]
            boxes=detections[:,:4]
            scores=detections[:,4]
            labels=detections[:,5]

        else:
            processed_image = self._preprocess_im(image)

            if len(processed_image.shape) == 3: #single image
                processed_image= processed_image.unsqueeze(0)

            with torch.no_grad():
                detections = self.detector(processed_image)[0]

            boxes = detections['boxes']
            scores = detections['scores']
            labels = detections['labels']

        #filter
        boxes, scores, labels = self._filter_detections(boxes, scores, labels)

        #normalizw
        normalized_boxes = self._normalize_roi_boxes(boxes, (width, height))

        # create the output detection tensor  [x1, y1, x2, y2, confidence, class]
        detection_results = torch.cat([
            normalized_boxes,
            scores.unsqueeze(1),
            labels.unsqueeze(1).float()
        ], dim=1)

        # embed rois
        embeddings = self._extract_features_from_roi(image, boxes)

        return detection_results, embeddings











