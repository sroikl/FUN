import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import List,Dict
from sklearn.cluster import KMeans
from itertools import combinations


def load_image(image_path: str) -> Image.Image:

    """Load an image from a file path."""
    image = Image.open(image_path)

    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def assign_colors_to_classes(class_names:List[str])-> dict:
    """chooses random colors for classes for visualizations"""
    n = len(class_names)
    colormap = plt.cm.tab10 if n <= 10 else plt.cm.tab20 if n <= 20 else plt.cm.viridis
    colors = [colormap(i / max(1, n - 1)) for i in range(n)]
    return dict(zip(class_names, colors))

def denormalize_bbox(bbox: torch.Tensor,width:int,height:int):
    x1, y1, x2, y2 = bbox[:4].tolist()
    x1 *= width
    y1 *= height
    x2 *= width
    y2 *= height

    return x1, y1, x2, y2

def visualize_detection_results(image:Image.Image,boxes:torch.Tensor,scores:torch.Tensor,root_save_path:str,class_names:List[str],save_results:bool)-> None:
    """

    :param image: the main image we want to visualize
    :param boxes: normalized boxes [x1, y1, x2, y2]
    :param scores: the predicted confidence scores
    :param labels: class labels
    :param class_names: names of the classes
    :return: PIL Image with visualization
    """

    np_img= np.array(image)
    width, height = np_img.shape[:2]

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(np_img)

    colors_dict= assign_colors_to_classes(class_names)

    #iterate over all the bboxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = denormalize_bbox(boxes[i], width, height)
        score = scores[i].item()
        class_name = class_names[i]
        color = colors_dict.get(class_name)

        #draw rect
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=2
        )
        ax.add_patch(rect)

        #annotate
        ax.text(
            x1,
            y1 - 5,
            f"{class_name}: {score:.2f}",
            color=color,
            fontsize=10,
            backgroundcolor='white'
        )

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if save_results:
        plt.savefig('{}/detections.png'.format(root_save_path))

    plt.close()


def visualize_embeddings(embeddings:torch.Tensor,class_names:List[str],class_centers:Dict,root_save_path:str,save_results:bool)-> None:

    """visuzalize embeddings with PCA"""


    stacked_centers_tensor = torch.stack(list(class_centers.values()), dim=0)
    embeddings= torch.concat((embeddings,stacked_centers_tensor), dim=0)
    embeddings_np = embeddings.cpu().numpy()
    pca = PCA(n_components=2)
    if embeddings_np.shape[0] <= 1:
        print("not enough embeddings")
        return

    embeddings_pca = pca.fit_transform(embeddings_np)

    num_centers= len(class_centers.values())
    center_embeddings_pca=embeddings_pca[-num_centers:,:]
    embeddings_pca= embeddings_pca[:-num_centers,:]

    classes_unique= np.unique(class_names)
    colors = assign_colors_to_classes(classes_unique)
    plt.figure(figsize=(10, 8))

    for i, (x, y) in enumerate(embeddings_pca):
        class_name = class_names[i]
        color = colors.get(class_name)

        plt.scatter(x, y, color=color, s=100)
        plt.text(x, y, class_name, fontsize=9)

    for i, (x, y) in enumerate(center_embeddings_pca):
        class_name = list(class_centers.keys())[i]
        color = colors.get(class_name)

        plt.scatter(x, y, color=color, s=300,marker='x')
        plt.text(x, y, '{}_CENTER'.format(class_name), fontsize=9)

    for class_name in classes_unique:
        plt.scatter([], [], color=colors[class_name], label=class_name)

    plt.legend()
    plt.xlabel('pca component 1')
    plt.ylabel('pca component 2')
    plt.title('embeddings visualization')

    if save_results:
        plt.savefig('{}/embedding_visualization_pca.png'.format(root_save_path))
    plt.close()

def calc_embeddings_distance(embeddings:torch.Tensor,class_names:List[str],root_save_path:str,save_results:bool)-> torch.Tensor:

    n_classes = len(class_names)

    if n_classes <= 1:
        print("Need at least 2 classes to calculate distances")
        return pd.DataFrame()

    class_centers = {}
    class_counts = {}

    classes_unique= np.unique(class_names)
    for cls in classes_unique:
        # Get indices for this class
        class_indices = [i for i, name in enumerate(class_names) if name == cls]
        class_counts[cls] = len(class_indices)

        class_embeddings = embeddings[class_indices].cpu().numpy()

        # i would like to have an accurate estimation of the cloud center to calculate distances rather than use the mean
        # but to use KMEANS i have to have more than 3  embedding vectors
        if len(class_indices) >= 3:
            kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
            kmeans.fit(class_embeddings)
            class_centers[cls] = torch.tensor(kmeans.cluster_centers_[0], device=embeddings.device)
        else:
            class_centers[cls] = torch.mean(embeddings[class_indices], dim=0)


    euclidean_distances = {}

    for cls1,cls2 in list(combinations(classes_unique, 2)):

            euclidean_distance = torch.norm(class_centers[cls1] - class_centers[cls2]).item()
            euclidean_distances['{}_{}'.format(cls1,cls2)] = euclidean_distance

    euclidean_df = pd.DataFrame(data=euclidean_distances.values(),index=euclidean_distances.keys(),columns=['euclidean_distance'])

    if save_results:
        euclidean_df.to_csv('{}/euclidean_distances.csv'.format(root_save_path))

    print('\n\n Distance calculated \n')
    print(euclidean_df)

    return class_centers

