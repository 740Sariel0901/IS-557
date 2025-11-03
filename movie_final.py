import json
import re
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
from joblib import dump
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
# Data Preparation
# ----------------------------
def split_data(df: pd.DataFrame):
    """Split features and target into train/test sets."""
    X = df[['average_rating', 'release_year', 'country', 'genre', 'description']]
    y = df['rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# ----------------------------
# Modeling
# ----------------------------
def build_ridge_model(X_train, y_train):
    """Build a Ridge regression pipeline with preprocessing and cross-validation."""
    numeric_features = ['average_rating', 'release_year']
    categorical_features = ['country', 'genre']

    # Column transformer for preprocessing
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features),
        ('text', SentenceEmbedder(), ['description']),
        ('num_inter', NumericInteraction('average_rating', 'release_year'), numeric_features),
        ('num_cat_inter', NumericCategoricalInteraction('average_rating', 'country'),
         ['average_rating', 'country'])
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('ridge', Ridge())
    ])

    param_grid = {'ridge__alpha': np.logspace(-3, 1, 5)}
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_root_mean_squared_error')
    grid.fit(X_train, y_train)

    return grid, grid.best_params_['ridge__alpha']


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_model(y_test, y_pred):
    """Compute RMSE, MAE, R2 metrics."""
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2


# ----------------------------
# Visualization
# ----------------------------
def plot_predictions(X_test, y_test, y_pred, save_path="predictions_summary.png"):
    """Plot three evaluation plots vertically in one figure and save as PNG."""
    errors = y_pred - y_test
    residuals = y_test - y_pred

    fig, axs = plt.subplots(3, 1, figsize=(6, 18))

    # 1. Predicted vs Actual
    axs[0].scatter(y_test, y_pred, alpha=0.6)
    axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axs[0].set_xlabel('Actual Rating')
    axs[0].set_ylabel('Predicted Rating')
    axs[0].set_title('Predicted vs Actual')

    # 2. Error Distribution
    axs[1].hist(errors, bins=30, alpha=0.7)
    axs[1].set_xlabel('Prediction Error')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Error Distribution')

    # 3. Residuals vs Average Rating
    axs[2].scatter(X_test['average_rating'], residuals, alpha=0.5)
    axs[2].set_xlabel('Average Rating')
    axs[2].set_ylabel('Residuals')
    axs[2].set_title('Residuals vs Avg Rating')

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main(train_path):
    # Load data
    df = json_to_dataframe(train_path)

    # Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Train
    model, best_alpha = build_ridge_model(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse, mae, r2 = evaluate_model(y_test, y_pred)

    print("Measurements:")
    print(f"Best alpha: {best_alpha}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R2: {r2:.3f}")

    # Visualizations
    plot_predictions(X_test, y_test, y_pred)

    # Save model
    dump(model, "final_model.joblib")


if __name__ == "__main__":
    train_file = '/Users/ZYR/Desktop/IS 557/letterboxd_train.jsonl'
    main(train_file)

