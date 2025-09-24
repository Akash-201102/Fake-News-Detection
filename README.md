# Fake-News-Detection
A machine learning project for detecting fake news using NLP techniques. Implemented in Python with support for both Jupyter Notebook and Google Colab, making it easy to run, explore, and reproduce results.
Fake News Detection
This project is all about detecting whether a news article is real or fake using machine learning and Natural Language Processing (NLP). It’s designed to be simple, easy to understand, and works on both Jupyter Notebook and Google Colab, so anyone can try it out without much setup.

What This Project Does
Cleans and processes news text: removing stopwords, punctuation, and unnecessary characters.
Converts text into numerical features using TF-IDF or similar methods.
Trains machine learning models to classify news as real or fake.
Evaluates the model using metrics like Accuracy, Precision, Recall, and F1-score.
Works seamlessly in Jupyter Notebook and Google Colab.

Tools & Libraries
Python 3
Pandas & NumPy for data handling
Scikit-learn for machine learning
NLTK (or similar) for text processing
Jupyter Notebook / Google Colab as the environment

Project Structure
Data Folder (data/) – This contains the datasets we use for training and testing, like train.csv and test.csv.
Notebooks Folder (notebooks/) – The Jupyter Notebook version of the project is here (FakeNewsDetection.ipynb), where all the steps from preprocessing to model evaluation are included.
Colab Folder (colab/) – If you prefer Google Colab, this folder has a notebook ready to run (FakeNewsDetection_Colab.ipynb).
Source Code (src/) – This includes Python scripts for modular implementation:
preprocessing.py – functions for cleaning and preparing text.
feature_extraction.py – convert text into numerical features using TF-IDF or Count Vectorizer.
model_training.py – training and evaluating the machine learning models.
requirements.txt – lists all Python libraries needed to run the project.

How to Run
On Jupyter Notebook
Clone the repo:
git clone <repository_url>
cd fake-news-detection

Install required libraries:
pip install -r requirements.txt

Open the notebook:
jupyter notebook notebooks/FakeNewsDetection.ipynb

Run all cells step by step.

On Google Colab
Upload FakeNewsDetection_Colab.ipynb to Google Colab
Make sure the data/ folder is uploaded or connected from Google Drive.
Run the cells in order.
