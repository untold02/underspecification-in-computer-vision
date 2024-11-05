import os
import numpy as np
import shap
import contextlib
import pickle

class_labels = [str(i) for i in range(10)]

def predict_fn(images, model):
    images = images.astype(np.float32) / 255.0
    return model.predict(images, verbose=0)


def shap_expls(sample_list, model):
    shap_scores = []

    masker = shap.maskers.Image("inpaint_telea", (32,32,3))

    for idx, (image, label) in enumerate(sample_list):
        print("Iteration :", idx + 1)

        img = np.array(image) if not isinstance(image, np.ndarray) else image
        img = (img * 255).astype(np.uint8)
        img = np.expand_dims(img, axis=0)

        explainer = shap.Explainer(lambda x: predict_fn(x, model), 
                                    masker, 
                                    output_names=class_labels)
        shap_value = explainer(
            img,
            batch_size = 500,
            max_evals  = 1000,
            outputs    = shap.Explanation.argsort.flip[:1]  # Focus on the top prediction
        )
        shap_scores.append(shap_value)

    return shap_scores


def generate_and_save_explanations(sample_lists, models, save_dir):
    """
    Generate and save explanations for all models and all classes.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for class_index in range(len(sample_lists)):
        
        for model_index, model in enumerate(models):
            
            explanations = shap_expls(sample_lists[class_index], model)
            
            filename = f"explain_{class_index}_{model.name}_shap.pkl"
            file_path = os.path.join(save_dir, filename)

            with open(file_path, 'wb') as file:
                pickle.dump(explanations, file)
            
            print(f"Explanations saved at {file_path}")
