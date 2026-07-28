#Import Libraries
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load Dataset
data = pd.read_csv("movie_dataset.csv")

# Use only first 5000 reviews for faster execution
data = data.head(5000)
print("Dataset Loaded Successfully")
print(data.head())

#Download Stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
print("Stopwords Loaded")

#Text preprocessing
def preprocess(text):
    text = str(text).lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)
data["Clean_Text"] = data["review"].apply(preprocess)
print(data[["review","Clean_Text"]].head())

#Feature Extraction using TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data["Clean_Text"])
y = data["sentiment"]
print("TF-IDF Completed")

#Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("Train-Test Split Completed")


#Train Model
model = MultinomialNB()

model.fit(X_train, y_train)

print("Model Training Completed")


#Test Model
prediction = model.predict(X_test)
print("\nPredicted Sentiments:")
print(prediction)
print("\nAccuracy Score:")
print(accuracy_score(y_test, prediction))
print("\nClassification Report:")
print(classification_report(y_test, prediction))


#Predict New Review
new_review = [
    "This movie was fantastic and I really enjoyed watching it."
]
new_review = [preprocess(review) for review in new_review]
new_text = tfidf.transform(new_review)
result = model.predict(new_text)
print("\nPrediction:")
print(result)

#Predict Another Review

new_review2 = [
    "This was the worst movie I have ever watched."
]
new_review2 = [preprocess(review) for review in new_review2]
new_text2 = tfidf.transform(new_review2)
result2 = model.predict(new_text2)
print("\nPrediction:")
print(result2)
