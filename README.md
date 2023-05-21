# Virtual Question Answering(VQA)

A Multimodal project in  order to solve Virtual Question Answering(VQA) challenge, one such challenge which requires high-level scene interpretation from images combined with language modelling of relevant Q&A. Given an image and a natural language question about the image, the task is to provide an accurate natural language answer.


## Generic Model Architecture ![Logo](https://camo.githubusercontent.com/5d485ae752f73ada4573280e09581e5d442add97ce67bc9b2d290c1a1a94eb80/687474703a2f2f692e696d6775722e636f6d2f327a4a30396d512e706e67)






## VQA 2.0 Dataset

The Visual Question Answering 2.0 (VQA 2.0) dataset is a large-scale benchmark for testing the ability of machine learning models to answer natural language questions about images. It contains over 1 million images from the COCO dataset, paired with over 2 million question-answer pairs. The questions are open-ended and cover a wide range of topics, including object recognition, spatial reasoning, and common sense knowledge. The answers are diverse and can be either single-word or free-form text. The dataset was designed to be challenging, with a balanced distribution of questions that require different levels of visual and linguistic reasoning. The VQA 2.0 dataset has been widely used to evaluate the performance of state-of-the-art models in visual question answering and has spurred research in areas such as multimodal representation learning, attention mechanisms, and commonsense reasoning. This dataset is a real challenge itself.
## Experiments

Applied different experiments and approaches starting from simple models turning into applying transformers, mostly all of them are completed.

| Model                                   |  Status     | Accuracy  |
| :--------                               | :-------    | :-------  |
| `VGG19 + LSTM`                          | Completed   | 31.56 %   |
| `InceptionV3 + LSTM`                    | Completed   | 37.2 %    |
| `InceptionV3 + GRU`                     | Completed   | 42.78 %   |
| `EffnetB2 + Bert`                       | Unfinished  | --        |
| `Vision Language Transformers(ViLT) `   | Completed   | 72.04 %   |

## How to Install and Run?

### Python environment requirements

```
  pip install -r requirements.txt
```

### Data upload

VQA 2.0 dataset is a huge dataset so in order to freely use it run the `Upload_VQA_Kaggle.ipynb` file to upload the entire dataset and use it freely on kaggle.


### Images Features Extraction

Upon which pre-trained model you are going to use you will have to run it's alternative feature extraction notebook. For example if you are going to use inception model you will have to run `vqa-image-features-inceptionv3.ipynb` this notebook uses a data loader to preprocess the 200 000 images file by file and batch by batch in order to extract the images features using the Pre-trained InceptionV3 model on ImageNet dataset and saves the extracted features in a pickle file, also it mapes the features to it's alternative images ID.

**Note:** if you are using ViLT model you don't have to extract image features as it loads by giving you model weights


### Train and Evaluate Models

Pass the preprocessed data to the model, then compile and then evaluate your model and save the results.

### Deployment Demo

Matching the results of the previous files to the deployment notebook you have 2 options:

-   **Custom Model:** Apply the demo to a custom model.

-   **Pre-trained Model:** Apply the demo to a full pre-trained model.

**Note:** each section is seprated in the `Deployment_Demo.ipynb` notebook.
## Experiments

Applied different experiments and approaches starting from simple models turning into applying transformers, mostly all of them are completed.

| Model                                   |  Status     | Accuracy  |
| :--------                               | :-------    | :-------  |
| `VGG19 + LSTM`                          | Completed   | 31.56 %   |
| `InceptionV3 + LSTM`                    | Completed   | 37.2 %    |
| `InceptionV3 + GRU`                     | Completed   | 42.78 %   |
| `EffnetB2 + Bert`                       | Unfinished  | --        |
| `Vision Language Transformers(ViLT) `   | Completed   | 72.04 %   |
