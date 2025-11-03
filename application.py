import json
import re
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

from joblib import load
import warnings

warnings.filterwarnings("ignore")  # suppress warnings for cleaner output


# ----------------------------
# Custom Transformers
# ----------------------------
class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """Convert text column into embeddings using SentenceTransformer."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)  # load once

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        return self.model.encode(X.tolist(), show_progress_bar=False)


class NumericInteraction(BaseEstimator, TransformerMixin):
    """Compute element-wise multiplication of two numeric columns."""

    def __init__(self, col1, col2):
        self.col1 = col1
        self.col2 = col2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.col1]].values * X[[self.col2]].values


class NumericCategoricalInteraction(BaseEstimator, TransformerMixin):
    """Compute numeric × categorical interaction using one-hot encoding."""

    def __init__(self, numeric_col, categorical_col):
        self.numeric_col = numeric_col
        self.categorical_col = categorical_col
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def fit(self, X, y=None):
        self.ohe.fit(X[[self.categorical_col]])
        return self

    def transform(self, X):
        num = X[[self.numeric_col]].values
        cat = self.ohe.transform(X[[self.categorical_col]])
        return num * cat


# ----------------------------
# Data Loading
# ----------------------------
def json_to_dataframe(path: str) -> pd.DataFrame:
    """Load JSONL dataset and parse HTML to extract features."""
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            for link, content in data.items():
                rating = content.get("rating")
                html_text = content.get("html", "")
                soup = BeautifulSoup(html_text, "html.parser")

                # Extract average_rating
                meta_tag = soup.find("meta", {"name": "twitter:data2"})
                if meta_tag:
                    match = re.search(r"[\d.]+", meta_tag.get("content", ""))
                    average_rating = float(match.group()) if match else np.nan
                else:
                    average_rating = np.nan

                # Extract release_year
                release_year = None
                for a in soup.find_all("a"):
                    if re.search(r'/films/year/\d{4}/', a.get("href", "")):
                        try:
                            release_year = int(a.get_text(strip=True))
                        except:
                            release_year = np.nan
                        break
                if release_year is None:
                    release_year = np.nan

                # Extract country
                country = None
                for a in soup.find_all("a"):
                    if re.search(r'/films/country/', a.get("href", "")):
                        country = a.get_text(strip=True)
                        break

                # Extract description
                description_tag = soup.find("meta", {"name": "description"}) or \
                                  soup.find("meta", {"property": "og:description"})
                description = description_tag.get("content", "") if description_tag else ""

                # Extract genres
                genres = [a.get_text(strip=True) for a in soup.find_all("a")
                          if re.search(r'/films/genre/', a.get("href", ""))]
                genre_str = ", ".join(genres) if genres else ""

                rows.append({
                    "link": link,
                    "rating": rating,
                    "average_rating": average_rating,
                    "release_year": release_year,
                    "country": country,
                    "description": description,
                    "genre": genre_str
                })

    return pd.DataFrame(rows)



# ----------------------------
# Display Predictions
# ----------------------------
def display_predictions(model, df):
    """Predict ratings for all rows and return as DataFrame."""
    results = []

    for row in df.itertuples(index=False):
        X_input = pd.DataFrame([[
            row.average_rating,
            row.release_year,
            row.country,
            row.genre,
            row.description
        ]], columns=['average_rating', 'release_year', 'country', 'genre', 'description'])

        y_pred = model.predict(X_input)
        results.append((row.link, round(y_pred[0], 1)))

    return pd.DataFrame(results, columns=['link', 'predicted_rating'])



# ----------------------------
# Main
# ----------------------------
def main(train_path, model_file):
    # Load data
    df = json_to_dataframe(train_path)

    # Get model
    model = load(model_file)

    # Display predictions
    predictions_df = display_predictions(model, df)
    print("Predictions:")
    for _, row in predictions_df.iterrows():
        print(f"{row['link']} {row['predicted_rating']}")

    return model, predictions_df


if __name__ == "__main__":
    train_file = '/Users/ZYR/Desktop/IS 557/letterboxd_train.jsonl'
    model_file = '/Users/ZYR/Desktop/IS 557/final_model.joblib'
    main(train_file, model_file)

