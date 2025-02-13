import tensorflow as tf
import numpy as np

class_labels = {
    0 : 'cat',
    1 : 'dog',
}
class_labels = list(class_labels.values())

def get_sample(class_index, dataset):
    sample_list = []

    for images, labels in dataset.unbatch():
        if labels.numpy() == class_index:
            sample_list.append((images.numpy(), labels.numpy()))
    return sample_list

def get_predictions(models, dataset):
    predictions = []
    index = 0

    for images, labels in dataset:
        correct = True
        
        for model in models:
            pred       = model.predict(np.expand_dims(images, axis=0), verbose=0)
            pred_label = class_labels[1] if pred >= 0.5 else class_labels[0]
            # true_label = class_labels[np.argmax(labels)]
            true_label = class_labels[labels]

            if pred_label != true_label:
                correct = False
                print(f"Wrong Prediction at index {index} by {model.name}: Predicted {pred_label}, True {true_label}")
                break

        predictions.append((images, labels, correct))
        index += 1

    return predictions

def filter_correct_predictions(predictions):
   return [(image, label) for image, label, correct in predictions if correct]
