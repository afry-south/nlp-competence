import numpy as np
import pandas as pd
import regex as re
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing punctuation
    - Lowering text
    """

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)  # Extra: Is regex needed? Other ways to accomplish this.
    text = re.sub(r"\"", "", text)
    # replace all non alphanumeric with space
    text = re.sub(r"\W+", " ", text)
    # text = re.sub(r"<.+?>", " ", text) # <br></br>hej<br></br>

    # Extra: How would we go ahead and remove HTML? Time to learn some Regex!

    return text.strip().lower()


def load_train_test_quora_data():
    """
    Loads the Quora train/test datasets from a folder path.

    Returns:
    train/test datasets as pandas dataframes.
    """
    import csv
    data = {}
    data["train"] = []
    data["test"] = []

    with open('data/quora/train.csv', 'r') as f:
        questions = [record for record in csv.reader(f, delimiter=',', quotechar='"')][1:]
        # random.shuffle(questions)
        split = int(0.8 * len(questions))

        for question in questions[:split]:
            data["train"].append([question[1], question[2]])
        for question in questions[split:]:
            data["test"].append([question[1], question[2]])

    # We shuffle the data to make sure we don't train on sorted data. This results in some bad training.
    np.random.shuffle(data["train"])
    data["train"] = pd.DataFrame(data["train"],
                                 columns=['text', 'sentiment'])

    np.random.shuffle(data["test"])
    data["test"] = pd.DataFrame(data["test"],
                                columns=['text', 'sentiment'])

    return data["train"], data["test"]


train_data, test_data = load_train_test_quora_data()

# Transform each text into a vector of word counts
vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                             lowercase=False,
                             strip_accents='ascii')

training_features = vectorizer.fit_transform(train_data["text"])
test_features = vectorizer.transform(test_data["text"])

# Training
model = LinearSVC()
model.fit(training_features, train_data["sentiment"])

y_pred = model.predict(test_features)

# Evaluation
acc = accuracy_score(test_data["sentiment"], y_pred)
f1 = f1_score(test_data["sentiment"], y_pred, pos_label='1')
print("Accuracy on the Quora dataset: {:.2f}".format(acc * 100))
print("F1 on the Quora dataset: {:.2f}".format(f1 * 100))

# Accuracy on the Quora dataset: 95.49
# F1 on the Quora dataset: 60.83
