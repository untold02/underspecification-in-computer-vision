import lime
import shap
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import numpy as np
import os
import pickle
import contextlib
import gc

class_labels = {
    0 : 'cat',
    1 : 'dog',
}
class_labels = list(class_labels.values())    # Cats_Vs._Dogs class labels

# create a LimeImageExplainer
class LIMEExplainer:
    def __init__(self):
        self.explainer = lime_image.LimeImageExplainer(verbose=False, random_state=42)
        self.segmenter = SegmentationAlgorithm("slic", 
                                                n_segments  = 50,               # For finer or coarser segmentation
                                                compactness = 10,               # Controls the balance between color similarity and spatial proximity
                                                sigma       = 1                 # Gaussian smoothing for better boundaries
                                              )

    def prediction_fn(self, images, model):
        images = np.array(images) if not isinstance(images, np.ndarray) else images

        if len(images.shape) == 3:                                              # if it is a single image case
            images = images.reshape(-1, 224, 224, 3)
        elif len(images.shape) == 4:                                            # if it is a batch case 
            if images.shape[1:] != (224, 224, 3):
                raise ValueError("Expected images with shape (224, 224, 3)")
            
        predictions = model.predict(images, verbose=0)

        # ensure binary predictions are in [P(class 0), P(class 1)] format
        if predictions.shape[1] == 1:                                           # if the model outputs a single probability value
            predictions = np.hstack([1 - predictions, predictions])             # convert to [P(class 0), P(class 1)]

        return predictions
    
    # Generate explanation for a single image instance
    def explain_aninstance(self, image, model, no_of_samples=1000):
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):                                 # Suppress unnecessary output
                explanation = self.explainer.explain_instance(image,
                                                        lambda x: self.prediction_fn(x, model),
                                                        top_labels  = 5,
                                                        hide_color  = 0,
                                                        num_samples = no_of_samples,
                                                        segmentation_fn = self.segmenter)
                
                temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                            positive_only = True,
                                                            num_features  = 5,
                                                            hide_rest     = True)
                
                return temp, mask, explanation

    # Generate explanations for all instances in the instance_list (per class)        
    def explanation_fn(self, instance_list, model, num_of_perturbations):
        temp_list  = []
        mask_list  = []
        expls_list = []

        for idx, (image, label) in enumerate(instance_list):
            print("Iteration :", idx+1)

            img = np.array(image)
            img = img.astype('double')

            explanation = self.explainer.explain_instance(img,
                                                lambda x: self.prediction_fn(x, model),
                                                hide_color  = 0,
                                                num_samples = num_of_perturbations)
            
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                        positive_only = True,
                                                        num_features  = 5,
                                                        hide_rest     = True)
            temp_list.append(temp)
            mask_list.append(mask)
            expls_list.append(explanation)

        return temp_list, mask_list, expls_list
    
    def generate_and_save_explanations(self, instance_list, models, save_dir, num_of_perturbations=1000):
        """
        Generate and save explanations for all models and all classes.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for class_index in range(len(instance_list)):
            for model in models:

                explanations = self.explanation_fn(instance_list[class_index][:50], model, num_of_perturbations)    # Generate explanations for the first 50 instances
                
                filename     = f"explain_{class_labels[class_index]}_{model.name}.pkl"
                file_path    = os.path.join(save_dir, filename)

                with open(file_path, 'wb') as file:
                    pickle.dump(explanations, file)

                del explanations
                gc.collect()




# create a SHAPExplainer
class SHAPExplainer:
    def __init__(self):
        self.masker = shap.maskers.Image("inpaint_telea", (224,224,3))

    def predict_fn(self, images, model):
        images = images.astype(np.float32) / 255.0
        preds  = model.predict(images, verbose=0)

        if preds.shape[1] == 1:
            # Convert to probability of each class [P(class 0), P(class 1)]
            preds = np.hstack([1 - preds, preds])
        return preds
    
    # Generate shap explanation for a single image instance
    def explain_aninstance(self, image, model):
        img = np.array(image) if not isinstance(image, np.ndarray) else image                           # Convert image to numpy array
        img = (img * 255).astype(np.uint8)                                                              # Convert image to uint8
        img = np.expand_dims(img, axis=0)                                                               # Add batch dimension, resulting shape: (1, 224, 224, 3)

        # Create an explainer object
        explainer = shap.Explainer(lambda x : self.predict_fn(x, model), 
                           self.masker, 
                           output_names=class_labels)
        
        # Generate SHAP values
        shap_value = explainer(
                            img,
                            batch_size = 500,                                                           # Batch size for the model
                            max_evals  = 1000,                                                          # Maximum number of evaluations for the model
                            outputs    = shap.Explanation.argsort.flip[:1])                             # Focus on the top 1 prediction
        return shap_value
    
    # Generate shap explanations for all instances in the instance_list (per class)
    def explanation_fn(self, instance_list, model):
        shap_scores = []

        for idx, (image, label) in enumerate(instance_list):
            print("Iteration :", idx+1)

            shap_value = self.explain_aninstance(image, model)
            shap_scores.append(shap_value)

        return shap_scores
    
    def generate_and_save_explanations(self, instance_lists, models, save_dir):
        """
        Generate and save explanations for all models and all classes.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for class_index in range(len(instance_lists)):
            for model in models:
                explanations = self.explanation_fn(instance_lists[class_index][:50], model)   # Generate explanations for the first 50 instances
                filename     = f"explain_{class_labels[class_index]}_{model.name}_shap.pkl"
                file_path    = os.path.join(save_dir, filename)

                with open(file_path, 'wb') as file:
                    pickle.dump(explanations, file)