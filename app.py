# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter('ignore')

app = Flask(__name__)

# --- Global Variables and Data Loading ---
# IMPORTANT: Ensure 'Reviews.csv' is in the same directory as app.py,
# or update this path to its correct location.
DATA_PATH = 'Reviews.csv'

df_final = None
pivot_df = None
preds_df = None
# New global variable to store the mapping from UserId (string) to its integer positional index (iloc)
user_id_to_iloc_idx = None

def load_and_preprocess_data():
    """
    Loads the dataset and performs all necessary preprocessing steps
    as done in the Jupyter notebook.
    This function will be called once when the Flask app starts.
    """
    global df_final, pivot_df, preds_df, user_id_to_iloc_idx

    print("Loading and preprocessing data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found. Please ensure the CSV file is in the correct directory.")
        return False

    # Dropping the columns as per notebook
    df = df.drop(['Id', 'ProfileName','Time','HelpfulnessNumerator','HelpfulnessDenominator','Text','Summary'], axis = 1)

    # Filter users who have given 50 or more ratings
    counts = df['UserId'].value_counts()
    df_final = df[df['UserId'].isin(counts[counts >= 50].index)]

    # Create the pivot table for Collaborative Filtering
    # UserId is kept as the index directly here
    pivot_df = pd.pivot_table(df_final, index=['UserId'], columns='ProductId', values="Score")
    pivot_df.fillna(0, inplace=True)

    # Create the mapping from UserId (string) to its integer positional index (iloc_idx)
    # This is crucial for correctly linking user IDs to SVD prediction rows,
    # as svds operates on the underlying numpy array, which has integer indexing.
    user_id_to_iloc_idx = {user_id: idx for idx, user_id in enumerate(pivot_df.index)}

    # Perform SVD
    # Convert pivot_df.values to a sparse matrix for svds.
    # The .values will give us a numpy array without the UserId index,
    # so its rows will correspond to the integer indices (0, 1, 2...)
    pivot_sparse = csr_matrix(pivot_df.values)
    
    # Ensure k is less than min(rows, cols)
    k_value = min(50, pivot_sparse.shape[0] - 1, pivot_sparse.shape[1] - 1)
    if k_value <= 0:
        print("Error: Not enough data for SVD. Check the number of users or products after filtering.")
        return False

    U, sigma, Vt = svds(pivot_sparse, k=k_value)
    sigma = np.diag(sigma)

    # Get all user predicted ratings
    # preds_df will have a default integer index (0, 1, 2...), corresponding
    # to the original row order (iloc) of the pivot_df values. Its columns are ProductIds.
    preds_df = pd.DataFrame(np.dot(np.dot(U, sigma), Vt), columns=pivot_df.columns)

    print("Data preprocessing complete.")
    return True

# --- Recommendation Functions ---

def recommend_popularity(num_recommendations=5):
    """
    Generates popularity-based recommendations.
    """
    if df_final is None:
        return []

    train_data_grouped = df_final.groupby('ProductId').agg({'UserId': 'count'}).reset_index()
    train_data_grouped.rename(columns={'UserId': 'score'}, inplace=True)
    train_data_sort = train_data_grouped.sort_values(['score', 'ProductId'], ascending=[0, 1])
    popularity_recommendations = train_data_sort.head(num_recommendations)

    recommended_products = popularity_recommendations['ProductId'].tolist()
    return recommended_products

def recommend_svd(user_id, num_recommendations=10):
    """
    Generates SVD-based collaborative filtering recommendations for a given user.
    """
    global user_id_to_iloc_idx # Access the global mapping

    if pivot_df is None or preds_df is None or user_id_to_iloc_idx is None:
        return []

    try:
        # Get the integer positional index (iloc_idx) for the given user_id
        if user_id not in user_id_to_iloc_idx:
            return f"User ID '{user_id}' not found in the dataset with sufficient ratings (min 50 ratings required)."

        user_iloc_idx = user_id_to_iloc_idx[user_id]

        # Get the user's actual ratings (from pivot_df using its original UserId string index)
        # And their predicted ratings (from preds_df using the integer iloc_idx)
        sorted_user_ratings = pivot_df.loc[user_id].sort_values(ascending=False)
        sorted_user_predictions = preds_df.iloc[user_iloc_idx].sort_values(ascending=False)

        # Concatenate for comparison
        temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1)
        temp.index.name = 'Recommended Items'
        temp.columns = ['user_ratings', 'user_predictions']

        # Filter out items the user has already rated and sort by predicted rating
        temp = temp.loc[temp.user_ratings == 0]
        temp = temp.sort_values('user_predictions', ascending=False)

        recommended_products = temp.head(num_recommendations).index.tolist()
        return recommended_products
    except Exception as e:
        print(f"Error during SVD recommendation for user {user_id}: {e}")
        return []

# --- Flask Routes ---

@app.route('/')
def index():
    """
    Renders the main input form.
    """
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """
    Handles the recommendation request from the form.
    """
    user_id = request.form.get('user_id', '').strip()
    try:
        num_recommendations = int(request.form.get('num_recommendations', 5))
    except ValueError:
        num_recommendations = 5 # Default if invalid input

    model_type = request.form.get('model_type')

    recommendations = []
    error_message = None

    if model_type == 'popularity':
        recommendations = recommend_popularity(num_recommendations)
        if not recommendations:
            error_message = "Could not generate popularity recommendations. Data might not be loaded or processed correctly."
    elif model_type == 'collaborative':
        if not user_id:
            error_message = "Please enter a User ID for Collaborative Filtering."
        else:
            recommendations = recommend_svd(user_id, num_recommendations)
            if isinstance(recommendations, str): # Check if it's an error message from recommend_svd
                error_message = recommendations
                recommendations = [] # Clear recommendations if it's an error
            elif not recommendations:
                # This specific message for collaborative filtering after user ID check
                error_message = f"No collaborative filtering recommendations found for User ID '{user_id}'. This user might not have enough ratings or there are no unrated items to recommend."
    else:
        error_message = "Please select a recommendation model type."

    return render_template('recommendations.html',
                           user_id=user_id if model_type == 'collaborative' else 'N/A (Popularity)',
                           num_recommendations=num_recommendations,
                           model_type=model_type,
                           recommendations=recommendations,
                           error_message=error_message)

# --- App Initialization ---
if __name__ == '__main__':
    # Load data once when the application starts
    if load_and_preprocess_data():
        app.run(debug=True)
    else:
        print("Application could not start due to data loading/preprocessing errors.")
