import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df.dropna(subset=["Text"], inplace=True)
        self.vectorizer = CountVectorizer(stop_words="english")
        self.feature_matrix = self.vectorizer.fit_transform(self.df["Text"])

    def get_recommendations(self, input_text, top_n=10):
        input_vec = self.vectorizer.transform([input_text])
        similarities = cosine_similarity(input_vec, self.feature_matrix).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        return self.df.iloc[top_indices][["ProductId", "Text", "Score"]].to_dict(orient="records")
