### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch
import numpy as np

from model import create_gadgets_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ["Headphones","Laptop","Mobile"]

### 2. Model and transforms preparation ###

# Create Gadget model
gadget, gadget_transforms = create_gadgets_model(
    num_classes=3, # len(class_names) would also work
)

# Load saved weights
gadget.load_state_dict(
    torch.load(
        f="Gadgets_model_save.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    threshold=0.87
    
    # Transform the target image and add a batch dimension
    img = gadget_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    gadget.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(gadget(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)   
    y=(pred_probs>threshold).float()
    pred_prob=y.long()
    pred_prob=pred_prob.numpy()
    x=np.count_nonzero(pred_prob)
    if x==0:
      pred_labels_and_probs={"Unknown Images found...Please Provide Images of Headphone,Mobile or laptop":0}
    else:
      pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}     
    
            
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings
title = "Gadgets Classifier 📱🖥🎧"
description = "An  computer vision model to classify images of Gadgets as Headphone, Laptop and Mobile."
article = "Created by Vishal Jadhav (www.linkedin.com/in/vishaljadhav1855)"



# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    examples=["Headphones17.jpg","Image_16.jpg","Image_20.jpg"],
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()
