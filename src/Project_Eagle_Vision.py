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
    return model


@st.cache()
def load_index_to_label_dict(
        path: str = 'src/index_to_class_label.json'
        ) -> dict:
    """Retrieves and formats the
    index to class label
    lookup dictionary needed to
    make sense of the predictions.
    When loaded in, the keys are strings, this also
    processes those keys to integers."""
    with open(path, 'r') as f:
        index_to_class_label_dict = json.load(f)
    index_to_class_label_dict = {
        int(k): v for k, v in index_to_class_label_dict.items()}
    return index_to_class_label_dict


def load_files_from_s3(
        keys: list,
        bucket_name: str = 'bird-classification-bucket'
        ) -> list:
    """Retrieves files anonymously from my public S3 bucket"""
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3_files = []
    for key in keys:
        s3_file_raw = s3.get_object(Bucket=bucket_name, Key=key)
        s3_file_cleaned = s3_file_raw['Body'].read()
        s3_file_image = Image.open(BytesIO(s3_file_cleaned))
        s3_files.append(s3_file_image)
    return s3_files


@st.cache()
def load_s3_file_structure(path: str = 'src/all_image_files.json') -> dict:
    """Retrieves JSON document outining the S3 file structure"""
    with open(path, 'r') as f:
        return json.load(f)


@st.cache()
def load_list_of_images_available(
        all_image_files: dict,
        image_files_dtype: str,
        bird_species: str
        ) -> list:
    """Retrieves list of available images given the current selections"""
    species_dict = all_image_files.get(image_files_dtype)
    list_of_files = species_dict.get(bird_species)
    return list_of_files


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
    index_to_class_label_dict = load_index_to_label_dict()
    all_image_files = load_s3_file_structure()
    types_of_birds = sorted(list(all_image_files['test'].keys()))
    types_of_birds = [bird.title() for bird in types_of_birds]

    st.title('Welcome To Tremor Power Quantification For Spiral Drawing!')
    instructions = """
        Upload your own spiral drawing image.
        The image you upload will be fed
        through the Deep Neural Network in real-time
        and the output will be displayed to the screen.
        """
    st.write(instructions)

    file = st.file_uploader('Upload An Image')

    if file:  # if user uploaded file
        img = Image.open(file).convert('RGB')
        # rgb_img = np.array(img.resize((224, 224)))
        prediction = predict(img, model)
        prediction = float(prediction.numpy())

        st.title("Here is the image you've selected")
        resized_image = img.resize((336, 336))
        st.image(resized_image)
        print(prediction)
        st.title("The predicted tremor level is {:.4f}".format(prediction))

