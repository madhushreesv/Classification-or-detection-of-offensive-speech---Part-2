import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import os, pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
from nltk.corpus import wordnet as wn
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, wordpunct_tokenize
import gensim
import gensim
from gensim import corpora
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.preprocessing import LabelEncoder
import joblib
import tensorflow as tf
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Check if eager execution is already enabled
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import nltk
import copy
from nltk.corpus import wordnet as wn
from tqdm import tqdm


def read_csv_dataset(file_name):
    """
    Read csv file and return data and print head of dataframe.

    Args:
        file_name: string

    Returns:
        panda dataframe
    """

    ##########################################################################
    #                     TODO: Implement this function                      #
    ##########################################################################
    # Replace "pass" statement with your code
    df = pd.read_csv(file_name)
    print("Read ", file_name)
    print(df.head())
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return df


def compute_performance(y_true, y_pred, split="test"):
    """
    prints different performance matrics like  Accuracy, Recall (macro), Precision (macro), and F1 (macro).
    This also display Confusion Matrix with proper X & Y axis labels.
    Also, returns F1 score

    Args:
        y_true: numpy array or list
        y_pred: numpy array or list


    Returns:
        float
    """

    ##########################################################################
    #                     TODO: Implement this function                      #
    ##########################################################################
    # Replace "pass" statement with your code
    print("Computing different preformance metrics on", split, " set of Dataset")
    f1score = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)

    print("F1 Score(macro): ", f1score)
    print("Accuracy: ", acc)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return f1score


import os
import pickle


def save_model(model, vectorizer, model_dir):
    """
    Save the model and vectorizer to the specified directory.

    Parameters:
        model (object): The model to be saved.
        vectorizer (object): The vectorizer used for feature extraction.
        model_dir (str): The directory where the model and vectorizer files will be saved.

    Returns:
        tuple: A tuple containing the file paths of the saved model and vectorizer.
    """
    # Save the model to disk
    model_file = os.path.join(model_dir, "model.sav")
    pickle.dump(model, open(model_file, "wb"))
    print("Saved model to", model_file)

    # Save the vectorizer to disk
    vectorizer_file = os.path.join(model_dir, "vectorizer.sav")
    pickle.dump(vectorizer, open(vectorizer_file, "wb"))
    return model_file, vectorizer_file

def save_model_machine_learning(model, vectorizer, model_dir):
    """
    Save the ml+unsup model and vectorizer to the specified directory.

    Parameters:
        model (object): The hypertuned model to be saved.
        vectorizer (object): The vectorizer used for feature extraction.
        model_dir (str): The directory where the model and vectorizer files will be saved.

    Returns:
        tuple: A tuple containing the file paths of the saved model and vectorizer.
    """
    # Save the hypertuned model to disk
    model_file = os.path.join(model_dir, "machine_learning_model.sav")
    pickle.dump(model, open(model_file, "wb"))
    print("Saved model to", model_file)

    # Save the vectorizer to disk
    vectorizer_file = os.path.join(model_dir, "vectorizer.sav")
    pickle.dump(vectorizer, open(vectorizer_file, "wb"))
    print("Saved Vectorizer to", vectorizer_file)

    return model_file, vectorizer_file
def save_model_hypertuning(model, vectorizer, model_dir):
    """
    Save the hypertuned model and vectorizer to the specified directory.

    Parameters:
        model (object): The hypertuned model to be saved.
        vectorizer (object): The vectorizer used for feature extraction.
        model_dir (str): The directory where the model and vectorizer files will be saved.

    Returns:
        tuple: A tuple containing the file paths of the saved model and vectorizer.
    """
    # Save the hypertuned model to disk
    model_file = os.path.join(model_dir, "hypertuned_model.sav")
    pickle.dump(model, open(model_file, "wb"))
    print("Saved model to", model_file)

    # Save the vectorizer to disk
    vectorizer_file = os.path.join(model_dir, "vectorizer.sav")
    pickle.dump(vectorizer, open(vectorizer_file, "wb"))
    print("Saved Vectorizer to", vectorizer_file)

    return model_file, vectorizer_file


def load_model(model_file, vectorizer_file):
    """
    Load the trained model and vectorizer from the specified files.

    Parameters:
        model_file (str): File path of the trained model.
        vectorizer_file (str): File path of the vectorizer used for feature extraction.

    Returns:
        tuple: A tuple containing the loaded model and vectorizer.
    """
    model = pickle.load(open(model_file, "rb"))
    print("Loaded model from", model_file)

    vectorizer = pickle.load(open(vectorizer_file, "rb"))
    print("Loaded Vectorizer from", vectorizer_file)

    return model, vectorizer


def text_preprocessing(text):
    # Tokenization: Split the text into words
    tokens = word_tokenize(text)

    # Lowercasing: Convert all words to lowercase
    tokens_lowercased = [word.lower() for word in tokens]

    # Noise removal: Remove special characters
    table = str.maketrans("", "", string.punctuation)
    tokens_no_punct = [word.translate(table) for word in tokens_lowercased]

    # Stop words removal: Remove commonly used words
    stop_words = set(stopwords.words("english"))
    tokens_filtered = [word for word in tokens_no_punct if word not in stop_words]

    # Join the tokens back to form a preprocessed text
    preprocessed_text = " ".join(tokens_filtered)

    return preprocessed_text


def prepare_dataset(data, count_vectorizer=None, split="test"):
    """
    Prepare the dataset for training or testing by vectorizing the text data using TfidfVectorizer.

    Args:
        data: pandas DataFrame
        count_vectorizer: CountVectorizer object (optional, required only for 'test' split)
        split: string, 'train' or 'test'

    Returns:
        For 'train' split: Transformed data and fitted TfidfVectorizer
        For 'test' split: Transformed data
    """

    if split == "train":
        count_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        values = count_vectorizer.fit_transform(data["preprocessed_text"].values)
        return values, count_vectorizer
    else:
        values = count_vectorizer.transform(data["preprocessed_text"].values)
        return values


def augment_data(train_df):
    """
    Augment the training data by replacing offensive tweets with their synonyms.

    Parameters:
        train_df (pd.DataFrame): The input DataFrame containing the training data.

    Returns:
        pd.DataFrame: The augmented DataFrame with offensive tweets replaced by their synonyms.
    """
    new_tweets = []
    new_labels = []
    for _, row in train_df.iterrows():
        tweet = row["preprocessed_text"]
        label = row["label"]

        if label == 1:
            results = augment_tweet(tweet)
            new_tweets.extend(results)
            new_labels.extend([1 for _ in range(len(results))])

    train_df_augmented = pd.concat(
        [
            train_df,
            pd.DataFrame({"preprocessed_text": new_tweets, "label": new_labels}),
        ],
        ignore_index=True,
    )

    return train_df_augmented


def word_to_replace(word, cand):
    """
    Check if a word should be replaced with a candidate synonym.

    Parameters:
        word (str): The original word to be replaced.
        cand (str): The candidate synonym word.

    Returns:
        bool: True if the word should be replaced, False otherwise.
    """
    distinct = word not in cand[: cand.find(".")]
    one_word = "_" not in cand
    return distinct and one_word


def word_to_be_replaced(token, tag):
    """
    Check if a word should be replaced based on its POS tag.

    Parameters:
        token (str): The tokenized word.
        tag (str): The POS tag of the word.

    Returns:
        bool: True if the word should be replaced, False otherwise.
    """
    noun = tag in ("NN", "NNS")
    verb = tag in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ")
    adjective = tag in ("JJ", "JJR", "JJS")
    letters_only = token.isalpha()
    length = len(token) > 2
    return (noun or verb or adjective) and letters_only and length


def augment_tweet(tweet):
    """
    Augment a single tweet by replacing its offensive words with synonyms.

    Parameters:
        tweet (str): The original tweet.

    Returns:
        list of str: A list of augmented tweets with replaced offensive words.
    """
    pos_map = {
        "NN": "n",
        "NNS": "n",
        "VB": "v",
        "VBD": "v",
        "VBG": "v",
        "VBN": "v",
        "VBP": "v",
        "VBZ": "v",
        "JJ": "a",
        "JJR": "a",
        "JJS": "a",
    }

    pos_tags = nltk.pos_tag(tweet.split())
    to_replace = [
        (i, token, tag)
        for i, (token, tag) in enumerate(pos_tags)
        if word_to_be_replaced(token, tag)
    ]

    if len(to_replace) == 0:
        return []

    new_data = []
    for idx, word, tag in to_replace:
        candidates = [
            l
            for l in wn.synsets(word, pos=pos_map[tag])
            if word_to_replace(word, l.name())
        ]
        if len(candidates) > 0:
            r = candidates[0].name()
            replacement = r[: r.find(".")]

            new_tweet = copy.deepcopy(tweet)
            new_tweet = " ".join(
                [
                    replacement if i == idx else token
                    for i, token in enumerate(new_tweet.split())
                ]
            )
            new_data.append(new_tweet)

    return new_data


def balance_dataset(train_df):
    """
    Balance the dataset by randomly sampling tweets from both classes to match the count of the minority class.

    Parameters:
        train_df (pd.DataFrame): The input DataFrame containing the training data.

    Returns:
        pd.DataFrame: A new DataFrame with balanced data, containing an equal number of tweets from both classes.
    """
    class_counts = train_df["label"].value_counts()
    tweets_in_class_0 = class_counts[0]
    desired_count = tweets_in_class_0
    # Find the number of tweets in each class
    class_counts = train_df["label"].value_counts()

    # Find the number of tweets in the minority class (Class 1)
    minority_class_count = class_counts.min()

    # Find the number of tweets in the majority class (Class 0)
    majority_class_count = class_counts.max()

    # Determine the desired count for the minority class to make it equal to the majority class
    desired_minority_count = max(minority_class_count, desired_count)

    # Determine the desired count for the majority class to make it equal to the minority class
    desired_majority_count = max(majority_class_count, desired_count)

    # Get the indices of tweets belonging to each class
    minority_class_indices = train_df[train_df["label"] == 1].index
    majority_class_indices = train_df[train_df["label"] == 0].index

    # Randomly sample from the minority class indices to reach the desired count
    minority_class_indices_sampled = np.random.choice(
        minority_class_indices, desired_minority_count, replace=True
    )

    # Randomly sample from the majority class indices to reach the desired count
    majority_class_indices_sampled = np.random.choice(
        majority_class_indices, desired_majority_count, replace=True
    )

    # Concatenate the minority class indices and the sampled majority class indices
    balanced_indices = np.concatenate(
        [minority_class_indices_sampled, majority_class_indices_sampled]
    )

    # Shuffle the indices
    np.random.shuffle(balanced_indices)

    # Create a new DataFrame with balanced data
    balanced_train_df = train_df.loc[balanced_indices]

    return balanced_train_df



def create_feature_matrix(ldamodel, corpus):
    """
    Create feature matrices for training and validation data using the given LDA model.

    Parameters:
        ldamodel (gensim.models.ldamodel.LdaModel): Trained LDA model.
        corpus_train (list of list): Corpus of training data in Bag of Words format.
        corpus_val (list of list): Corpus of validation data in Bag of Words format.

    Returns:
        tuple: A tuple containing the feature matrices.
    """

    num_topics = ldamodel.num_topics

    # Prepare the feature matrix for training data
    topic_probabilities = np.zeros((len(corpus), num_topics))
    for i, doc in enumerate(corpus):
        for topic, prob in ldamodel.get_document_topics(doc):
            topic_probabilities[i, topic] = prob

    return topic_probabilities


def lemmatized_text(text):
    """
    Preprocess the input text by tokenizing, removing stopwords, and lemmatizing the words.

    Parameters:
        text (str): The input text to be preprocessed.

    Returns:
        list: A list of lemmatized words after tokenization and stopword removal.
    """
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Tokenize the text
    tokenized_text = word_tokenize(text)

    # Remove stopwords
    filtered_text = [word for word in tokenized_text if word not in stop_words]

    # Lemmatization
    lemmatized_text = [lemmatizer.lemmatize(word) for word in filtered_text]

    return lemmatized_text


def get_dominant_topic(ldamodel, corpus):
    """
    Perform Latent Dirichlet Allocation (LDA) topic modeling on a given corpus and extract the dominant topic for each document.

    Parameters:
        ldamodel: A trained LDA model obtained using a library like Gensim.
        corpus (list): A list of document-term matrices or Bag of Words representation of the corpus.

    Returns:
        list: A list of integers representing the dominant topic index for each document in the corpus.
    """
    topic_list = []
    for i, doc in enumerate(corpus):
        topic_probabilities = ldamodel.get_document_topics(doc)
        dominant_topic = max(topic_probabilities, key=lambda x: x[1])
        topic_list.append(dominant_topic[0])
    return topic_list


# Common function for preprocessing and topic modeling
def preprocess_and_topic_model(documents, num_topics):
    # Preprocess training documents
    lemmatized_documents = [lemmatized_text(doc) for doc in documents]

    # Create a dictionary and corpus from the documents
    dictionary = corpora.Dictionary(lemmatized_documents)
    gensim_corpus = [dictionary.doc2bow(doc) for doc in lemmatized_documents]

    # Build the LDA model for training data
    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus=gensim_corpus, num_topics=num_topics, id2word=dictionary, passes=15
    )

    # Get the dominant topic for each document.
    dominant_topics = get_dominant_topic(ldamodel, gensim_corpus)
    return ldamodel, gensim_corpus, dictionary, dominant_topics, lemmatized_documents


def save_label_encoder(label_encoder, filename):
    """
    Save the LabelEncoder to the specified file.

    Parameters:
        label_encoder (LabelEncoder): The fitted LabelEncoder to save.
        filename (str): The name of the file to save the LabelEncoder.

    Returns:
        None
    """
    with open(filename, "wb") as f:
        pickle.dump(label_encoder, f)

    print("Saved LabelEncoder to", filename)


import pickle


def load_label_encoder(filename):
    """
    Load the LabelEncoder from the specified file.

    Parameters:
        filename (str): The name of the file to load the LabelEncoder from.

    Returns:
        LabelEncoder: The loaded LabelEncoder.
    """
    with open(filename, "rb") as f:
        label_encoder = pickle.load(f)

    return label_encoder


def save_lda_model_and_dictionary(ldamodel, dictionary, model_file, dictionary_file):
    """
    Save the trained LDA model and dictionary to files.

    Args:
        ldamodel (gensim.models.LdaModel): Trained LDA model.
        dictionary (gensim.corpora.Dictionary): Dictionary used for LDA.
        model_file (str): File path to save the LDA model.
        dictionary_file (str): File path to save the dictionary.

    Returns:
        None
    """
    joblib.dump(ldamodel, model_file)
    joblib.dump(dictionary, dictionary_file)
    print(f"Saved LDA model to {model_file}")
    print(f"Saved dictionary to {dictionary_file}")


def calculate_dataset_details(train_df, test_df, val_df):
    """
    This function calculates the dataset details.
    Args:
        train_df: string
        test_df: string
        val_df: string

    Returns:
       returns percentage of samples for each class (Class A and Class B) in each dataset in a pandas dataframe.
    """
    # Calculate total samples in each dataset
    total_samples_train = len(train_df)
    total_samples_test = len(test_df)
    total_samples_val = len(val_df)

    # Calculate percentage of Class A and Class B in each dataset
    class_a_train = len(train_df[train_df["label"] == "NOT"])
    class_b_train = len(train_df[train_df["label"] == "OFF"])
    percent_class_a_train = (class_a_train / total_samples_train) * 100
    percent_class_b_train = (class_b_train / total_samples_train) * 100

    class_a_test = len(test_df[test_df["label"] == "NOT"])
    class_b_test = len(test_df[test_df["label"] == "OFF"])
    percent_class_a_test = (class_a_test / total_samples_test) * 100
    percent_class_b_test = (class_b_test / total_samples_test) * 100

    class_a_val = len(val_df[val_df["label"] == "NOT"])
    class_b_val = len(val_df[val_df["label"] == "OFF"])
    percent_class_a_val = (class_a_val / total_samples_val) * 100
    percent_class_b_val = (class_b_val / total_samples_val) * 100

    # Calculate "Original" instances
    total_samples_original = (
        total_samples_train + total_samples_val + total_samples_test
    )
    class_a_original = class_a_train + class_a_val + class_a_test
    class_b_original = class_b_train + class_b_val + class_b_test
    percent_class_a_original = (class_a_original / total_samples_original) * 100
    percent_class_b_original = (class_b_original / total_samples_original) * 100

    # Prepare the dataset details table as a dictionary
    dataset_details = {
        "Dataset": ["Original", "Train", "Valid", "Test"],
        "Total": [
            total_samples_original,
            total_samples_train,
            total_samples_val,
            total_samples_test,
        ],
        "% Class A": [
            f"{percent_class_a_original:.2f}",
            f"{percent_class_a_train:.2f}",
            f"{percent_class_a_val:.2f}",
            f"{percent_class_a_test:.2f}",
        ],
        "% Class B": [
            f"{percent_class_b_original:.2f}",
            f"{percent_class_b_train:.2f}",
            f"{percent_class_b_val:.2f}",
            f"{percent_class_b_test:.2f}",
        ],
    }

    return pd.DataFrame(dataset_details)


def prepare_dataset_nn(df, sample_flag=False, count_vectorizer=None):
    """
    Prepare the dataset for training a neural network.

    Parameters:
        PATH (str): The path where the dataset file is located.
        file_name (str): The name of the dataset file.
        sample_flag (bool): If True, only 10% of the data points will be used for simplicity and faster training.
        count_vectorizer (CountVectorizer, optional): An existing CountVectorizer instance. If not provided, a new one will be created.

    Returns:
        TensorDataset: A TensorDataset containing the input features and labels for training.
        int: The input size (number of features) of the dataset.
        CountVectorizer: The CountVectorizer instance used for vectorizing the text data.
    """
    data = df
    if sample_flag:
        # For sake of simplicity let's use only 20% data points
        # data = data.sample(frac=0.5).reset_index(drop=True)  # Shuffling dataset
        data = data  # considering entire datasets
    if count_vectorizer == None:
        count_vectorizer = CountVectorizer(stop_words="english", max_features=5000)
        values = count_vectorizer.fit_transform(
            data["preprocessed_text"].values
        )  # TODO: This is the best way to do this, because you need to use same vectorization menthod
    else:
        values = count_vectorizer.transform(data["preprocessed_text"].values)

    labels = data["label"].values

    # Convert into Tensor
    values = torch.tensor(values.toarray()).float()
    labels = torch.tensor(labels)

    dataset = TensorDataset(values, labels)
    input_size = values.shape[1]
    return dataset, input_size, count_vectorizer


import torch.nn as nn

class HateSpeechClassificationMLP(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, num_class, hidden_size1, hidden_size2, hidden_size3
    ):
        super(HateSpeechClassificationMLP, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)  # Remove sparse=True
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(embed_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()
        self.fc4.weight.data.uniform_(-initrange, initrange)
        self.fc4.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        h1 = self.fc1(embedded)
        a1 = self.activation(h1)
        h2 = self.fc2(a1)
        a2 = self.activation(h2)
        h3 = self.fc3(a2)
        a3 = self.activation(h3)
        h4 = self.fc4(a3)
        y = h4
        return y


import os
import pickle


def save_model_nn(model, tokenizer, model_dir):
    """
    Save the model, tokenizer to the specified directory.

    Parameters:
        model (object): The model to be saved.
        tokenizer (function): The tokenizer function used for text preprocessing.
        model_dir (str): The directory where the model and vectorizer files will be saved.

    Returns:
        tuple: A tuple containing the file paths of the saved model and tokenizer.
    """
    # Save the model to disk
    model_file = os.path.join(model_dir, "model.sav")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    print("Saved model to", model_file)

    # Save the tokenizer to disk
    tokenizer_file = os.path.join(model_dir, "tokenizer.sav")
    with open(tokenizer_file, "wb") as f:
        pickle.dump(tokenizer, f)
    print("Saved tokenizer to", tokenizer_file)

    return model_file, tokenizer_file


from torch.utils.data import Dataset


class HateSpeechClassificationMLPDataset(Dataset):
    def __init__(self, data_frame, tokenizer, vocab):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        text = self.data_frame.iloc[idx]["preprocessed_text"]
        label = self.data_frame.iloc[idx]["label"]
        tokenized_text = torch.tensor(
            self.vocab(self.tokenizer(text)), dtype=torch.int64
        )
        return tokenized_text, label


def predict(model, device, dataloader):
    """
    Predicts labels using a given model on a dataset using the specified device.

    Parameters:
        device (torch.device): The device (e.g., 'cuda', 'cpu') on which the model and data should be loaded.
        model (torch.nn.Module): The PyTorch model to be used for prediction.
        data_loader (torch.utils.data.DataLoader): The data loader containing the input data for prediction.

    Returns:
        tuple: A tuple containing two lists.
            - all_gt_labels (list): A list containing the ground truth labels for each input sample in the data_loader.
            - all_predict_labels (list): A list containing the predicted labels for each input sample in the data_loader.
    """
    model.eval()
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for label, text, offsets in dataloader:
            text, label = text.to(device), label.to(device)
            predicted_label = model(text, offsets)
            pred_labels.extend(predicted_label.argmax(dim=1).cpu().tolist())
            gt_labels.extend(label.cpu().tolist())

    return gt_labels, pred_labels


def collate_batch(batch):
    """
    Collates a batch of data for a DataLoader.

    Parameters:
        batch (list): A list of tuples containing (_text, _label) pairs.

    Returns:
        tuple: A tuple containing label_list, text_list, and offsets.
            - label_list (torch.Tensor): A tensor containing the labels of the batch.
            - text_list (torch.Tensor): A tensor containing the processed text data of the batch.
            - offsets (torch.Tensor): A tensor containing the offsets of the text data.
    """
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = _text
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets

# Function to evaluate the model
def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model's performance on the given dataloader using the specified criterion.

    The function operates in evaluation mode, where no gradients are computed during the process.
    It iterates through the dataloader, calculates the loss, and accumulates it to calculate the total loss.

    Parameters:
        model (object): The PyTorch model to be evaluated.
        dataloader (DataLoader): The DataLoader containing the data for evaluation.
        criterion (object): The loss function used to calculate the model's performance.
        device (str): The device (e.g., "cuda" or "cpu") on which the model and data are located.

    Returns:
        float: The average loss calculated over the evaluation dataset.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            text, label = text.to(device), label.to(device)
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_loss += loss.item()

    return total_loss / len(dataloader)

