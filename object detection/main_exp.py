import os
import argparse

from object_embedder import ObjectEmbedder
from detectors import detector_constructor
from feature_extractors import feature_extractor_constructor
from viz_utils import *

def run_experiment():


    parser = argparse.ArgumentParser(description='Object Embedder Demo')
    parser.add_argument('--data_dir', type=str, default='data' , help='directory of images')
    parser.add_argument('--threshold', type=float, default=0.9, help='detection confidence threshold')
    parser.add_argument('--feature_extractor', type=str, default='efficientnet', help='feature extractor name: [resnet50|efficientnet]')
    parser.add_argument('--detector_name', type=str, default='fasterrcnn', help='detctor name:[fasterrcnn|maskrcnn|yolo]')
    parser.add_argument('--save_results', type=bool, default=True, help='save results in a results directory')
    args = parser.parse_args()

    if not os.listdir(args.data_dir) or args.data_dir is None:
        print('No images found')
        return

    images= [load_image('{}/{}'.format(args.data_dir,x)) for x in os.listdir(args.data_dir)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #set up the tools
    detector,class_names = detector_constructor(args.detector_name)
    feature_extractor = feature_extractor_constructor(args.feature_extractor)

    #set up results path
    root_results_path= './results'
    if not os.path.exists(root_results_path):
        os.makedirs(root_results_path)

    #set up object embedding class
    embedder= ObjectEmbedder(detector= detector,
                             feature_extractor=feature_extractor,
                             detection_th=args.threshold,
                             device=device,
                             class_names=class_names,
                             detetor_name=args.detector_name)

    for image,image_name in zip(images,os.listdir(args.data_dir)):
        print('\n\nProcessing image: {}'.format(image_name))
        unique_save_path= '{}/{}_{}_{}'.format(root_results_path,image_name.split('.')[0],args.detector_name,args.feature_extractor)
        if not os.path.exists(unique_save_path):
            os.makedirs(unique_save_path)

        #get embeddings and detections
        detection_results, embeddings= embedder(image=image)

        boxes = detection_results[:,:4]
        scores = detection_results[:,4]
        classes = [class_names[x] for x in  detection_results[:,5].int().tolist()]

        #visualize results
        visualize_detection_results(image=image,
                                    boxes=boxes,
                                    scores=scores,
                                    class_names=classes,
                                    root_save_path=unique_save_path,
                                    save_results= args.save_results)

        class_centers= calc_embeddings_distance(embeddings=embeddings,
                                                class_names=classes,
                                                root_save_path=unique_save_path,
                                                save_results=args.save_results)

        visualize_embeddings(embeddings=embeddings,
                             class_names=classes,
                             class_centers=class_centers,
                             root_save_path=unique_save_path,
                             save_results= args.save_results)


    print('Done')
    return  detection_results,embeddings


if __name__ == '__main__':
    run_experiment()
