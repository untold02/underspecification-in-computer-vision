import numpy as np
import tensorflow as tf

def prediction_fn(images, model):
    # Convert images to numpy array
    images = np.array(images)

    # Check if the input is a single image
    if len(images.shape) == 3:  # Single image case
        images = images.reshape(-1, 32, 32, 3)
    # Check if the input is a batch of images
    elif len(images.shape) == 4:  # Batch case
        if images.shape[1:] != (32, 32, 3):
            raise ValueError("Expected images with shape (32, 32, 3)")

    # Get predictions from the model
    predictions = model.predict(images, verbose=0)

    return predictions
    
def get_sample(class_index, dataset):
    sample_list = []

    # Iterate through the dataset
    for images, labels in dataset.unbatch():
        # Check if the label matches the class index
        if np.argmax(labels, axis=0) == class_index: # labels are categorical
            sample_list.append((images, labels))
            
            # limit the number of samples
            # if len(sample_list) == num_of_samples:
            #     break

    return sample_list

def get_predictions(models, dataset):
    predictions = []
    index = 0

    # Iterate through the dataset
    for images, labels in dataset:
        correct = True
        # Check predictions for each model
        for model in models:
            pred = model.predict(tf.expand_dims(images, axis=0), verbose=0)
            pred_label = np.argmax(pred, axis=-1)
            true_label = np.argmax(labels.numpy(), axis=-1)

            # If any model makes a wrong prediction, set correct to False
            if pred_label != true_label:
                correct = False
                print(f"Wrong Prediction at index {index} by {model.name}: Predicted {pred_label}, True {true_label}")
                break

        # Append the result to the predictions list
        predictions.append((images, labels, correct))
        index += 1

    return predictions

def filter_correct_predictions(predictions):
    # Filter out only the correct predictions
    return [(image, label) for image, label, correct in predictions if correct]