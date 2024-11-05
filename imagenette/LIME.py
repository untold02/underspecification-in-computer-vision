import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
from lime.wrappers.scikit_image import SegmentationAlgorithm
import numpy as np
import os
import pickle
import contextlib
from labels import class_labels
import gc

# create a LimeImageExplainer
explainer = lime_image.LimeImageExplainer(verbose=False, random_state =42)

def prediction_fn(images, model):
    images = np.array(images)
    
    if len(images.shape) == 3:  #if it is a single image case
        images = images.reshape(-1, 224, 224, 3)
    elif len(images.shape) == 4:  #if it is a batch case
        if images.shape[1:] != (224, 224, 3):
            raise ValueError("Expected images with shape (224, 224, 3)")

    predictions = model.predict(images, verbose=0)

    return predictions


def explain_aninstance(image, model, no_of_samples = 1000):
    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f):
            explanation = explainer.explain_instance(image,
                                                    lambda x: prediction_fn(x, model),
                                                    top_labels  = 3,
                                                    hide_color  = 0,
                                                    num_samples = no_of_samples)
            
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                        positive_only = True,
                                                        num_features  = 5,
                                                        hide_rest     = True)
            

            return temp, mask, explanation

def explanation_fn(instance_list, model, num_of_perturbations):

    temp_list  = []
    mask_list  = []
    expls_list = []

    for idx, (image, label) in enumerate(instance_list):
        
        print("Iteration :", idx+1)

        img = np.array(image)
        img = img.astype('double')

        explanation = explainer.explain_instance(img,
                                            lambda x: prediction_fn(x, model),
                                            top_labels  = 5, 
                                            hide_color  = 0,
                                            num_samples = num_of_perturbations)
        
        print(f"Predicted Class : {np.argmax(prediction_fn(image, model))}")
        print(f"Explaining Class: {explanation.top_labels[0]}")

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                    positive_only = True, 
                                                    num_features  = 5, 
                                                    hide_rest     = True)

        temp_list.append(temp)
        mask_list.append(mask)
        expls_list.append(explanation)

    return temp_list, mask_list, expls_list


def generate_and_save_explanations(sample_list, models, num_of_perturbations, save_dir):
    """
    Generate and save explanations for all models and all classes.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for class_index in range(len(sample_list)):
        for model in models:
            
            explanations = explanation_fn(sample_list[class_index][:50], model, num_of_perturbations)
            
            filename     = f"explain_{class_labels[class_index]}_{model.name}.pkl"
            file_path    = os.path.join(save_dir, filename)

            with open(file_path, 'wb') as file:
                pickle.dump(explanations, file)

            del explanations
            gc.collect()
