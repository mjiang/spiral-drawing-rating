import json
from io import BytesIO
from PIL import Image
import os

import boto3
from botocore import UNSIGNED  # contact public s3 buckets anonymously
from botocore.client import Config  # contact public s3 buckets anonymously

import streamlit as st
import pandas as pd
import numpy as np
import torch

# from resnet_model import ResnetModel
from baseline_model import baseline

@st.cache()
def load_model(path: str = 'models/checkpoints/checkpoint_best.pth') -> baseline:
    """Retrieves the trained model and maps it to the CPU by default,
    can also specify GPU here."""
    model = baseline()
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


@st.cache()
def predict(
        img: Image.Image,
        model,
        ) -> list:
    """Transforming input image according to ImageNet paper
    The Resnet was initially trained on ImageNet dataset
    and because of the use of transfer learning, I froze all
    weights and only learned weights on the final layer.
    The weights of the first layer are still what was
    used in the ImageNet paper and we need to process
    the new images just like they did.

    This function transforms the image accordingly,
    puts it to the necessary device (cpu by default here),
    feeds the image through the model getting the output tensor,
    converts that output tensor to probabilities using Softmax,
    and then extracts and formats the top k predictions."""
    with torch.no_grad():
        formatted_predictions = model.prediction(img)
    return formatted_predictions


if __name__ == '__main__':
    model = load_model()
    st.title('Welcome To Spiral Drawing Test!')
    instructions = """
        Tremor is the most common movement disorder. The assessment of tremor is in general carried out in a very subjective way. The Archimedean spiral drawing is a standardized test designed to capture the frequency, amplitude and direction of a tremor for a quantitative assessment. To perform this test, grab a pen and a piece of paper, and draw the following Archimedean spiral. After that, take a picture of your spiral drawing and upload it, to assess your tremor severity.
        """
    st.write(instructions)
    st.image("https://fasikl.s3.us-west-2.amazonaws.com/videos/spiral.png")
 
    file = st.file_uploader('Upload your spiral drawing:')

    if file:  # if user uploaded file
        img = Image.open(file).convert('RGB')
        # rgb_img = np.array(img.resize((224, 224)))
        prediction = predict(img, model)
        prediction = float(prediction.numpy())

        st.title("Here is the drawing you've uploaded")
        resized_image = img.resize((336, 336))
        st.image(resized_image)
        print(prediction)
        severities = ['no', 'a slight', 'a mild', 'a moderate', 'a severe']
        rating = min(max(int(prediction + 0.5), 0), 4)
        st.title("Your score is {:.2f}. You have {} tremor!".format(prediction, severities[rating]))
