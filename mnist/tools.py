import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
from lime.wrappers.scikit_image import SegmentationAlgorithm
import numpy as np
import os
import pickle
import contextlib
import shap

# Class for generating image explanations using LIME
class LIMEExplainer:
    def __init__(self):
        # Initialize the LIME explainer and segmentation algorithm
        self.explainer = lime_image.LimeImageExplainer(verbose = False, random_state = 42)
        self.segmenter = SegmentationAlgorithm("slic", kernal_size = 3)
    


    def prediction_fn(self, images, model):
        # Convert images to numpy array
        images = np.array(images)

        # Check if images are single or batch
        if len(images.shape) == 3:                       # Single image case
            images = images.reshape(-1, 32, 32, 3)
        elif len(images.shape) == 4:                     # Batch case
            if images.shape[1:] != (32, 32, 3):
                raise ValueError("Expected images with shape (32, 32, 3)")
            
        # Get predictions
        predictions = model.predict(images, verbose = 0)

        return predictions
    


    def explain_aninstance(self, image, model, no_of_samples):
        # Generate explanation for a single image instance
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):         # Suppress unnecessary output
                img = np.array(image)
                img = img.astype('double')
                
                explanation = self.explainer.explain_instance(img,
                                                        lambda x: self.prediction_fn(x, model),
                                                        top_labels      = 3,
                                                        hide_color      = 0,
                                                        num_samples     = no_of_samples,
                                                        segmentation_fn = self.segmenter
                                                                )
                
                temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                            positive_only = True,
                                                            num_features  = 5,
                                                            hide_rest     = True)
                

                return temp, mask, explanation
            


    def explanation_fn(self, sample_list, model, num_of_perturbations):
        # Generate explanations for a list of image instances
        temp_list  = []
        mask_list  = []
        expls_list = []

        for idx, (image, label) in enumerate(sample_list):
            print("Iteration :", idx+1)

            img = np.array(image)
            img = img.astype('double')

            temp, mask, explanation = self.explain_aninstance(img, model, num_of_perturbations)
            temp_list.append(temp)
            mask_list.append(mask)
            expls_list.append(explanation)

        return temp_list, mask_list, expls_list
    


    def generate_and_save_explanations(self, sample_lists, models, save_dir, num_of_perturbations):
        # Generate and save explanations for all models and all classes
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for class_index in range(len(sample_lists)):
            
            for model_index, model in enumerate(models):
                
                expls     = self.explanation_fn(sample_lists[class_index], model, num_of_perturbations)
                
                filename  = f"explain_{class_index}_{model.name}_lime.pkl"
                file_path = os.path.join(save_dir, filename)

                with open(file_path, 'wb') as file:
                    pickle.dump(expls, file)
                
                print(f"Explanations saved at {file_path}")





# Class for generating image explanations using SHAP
class SHAPExplainer:
    def __init__(self):
        self.masker       = shap.maskers.Image("inpaint_telea", (32,32,3))                    # Masker to hide parts of the image
        self.class_labels = [str(i) for i in range(10)]                                       # MNIST classes: 0-9 

        def predict_fn(self, images, model):
            # Convert images to numpy array
            images = images.astype(np.float32) / 255.0

            return model.predict(images, verbose = 0)
        



    def explain_aninstance(self, image, model):
        # Convert image to numpy array
        img = np.array(image) if not isinstance(image, np.ndarray) else image
        img = (img * 255).astype(np.uint8)
        img = np.expand_dims(img, axis=0)

        explainer = shap.Explainer(lambda x: self.predict_fn(x, model), 
                                    self.masker,
                                    output_names = self.class_labels)
        
        shap_value = explainer(
            img,
            batch_size = 500,
            max_evals  = 1000,
            outputs    = shap.Explanation.argsort.flip[:1]  # Focus on the top prediction
        )

        return shap_value




    def explanation_fn(self, sample_list, model):
        # Generate SHAP explanations for a list of image instances
        shap_scores = []

        for idx, (image, label) in enumerate(sample_list):
            print("Iteration :", idx + 1)
            shap_value = self.explain_aninstance(image, model)
            shap_scores.append(shap_value)

        return shap_scores


 
        
    def generate_and_save_explanations(self, sample_lists, models, save_dir):
        # Generate and save SHAP explanations for all models and all classes
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for class_index in range(len(sample_lists)):
            
            for model_index, model in enumerate(models):
                
                explanations = self.explanation_fn(sample_lists[class_index], model)
                
                filename = f"explain_{class_index}_{model.name}_shap.pkl"
                file_path = os.path.join(save_dir, filename)

                with open(file_path, 'wb') as file:
                    pickle.dump(explanations, file)
                
                print(f"Explanations saved at {file_path}")