from labels import class_labels
import tensorflow as tf
import numpy as np

def get_sample(class_index, dataset):
    sample_list = []

    for images, labels in dataset.unbatch():
        if tf.argmax(labels, axis=0).numpy() == class_index:
            sample_list.append((images.numpy(), labels.numpy()))
    return sample_list

def get_predictions(models, dataset):
    predictions = []
    index = 0

    for images, labels in dataset:
        correct = True
        for model in models:
            pred = model.predict(np.expand_dims(images, axis=0), verbose=0)
            pred_label = class_labels[np.argmax(pred)]
            true_label = class_labels[np.argmax(labels)]

            if pred_label != true_label:
                correct = False
                break

        if not correct:
            print(f"Wrong Prediction at index {index} by {model.name}: Predicted {pred_label}, True {true_label}")
            
        predictions.append((images, labels, correct))
        index +=1

    return predictions

def filter_correct_predictions(predictions):
   return [(image, label) for image, label, correct in predictions if correct]