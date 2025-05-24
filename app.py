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
DATA_PATH = 'Reviews.csv'

df_final = None
pivot_df = None
preds_df = None
user_id_to_iloc_idx = None
product_avg_scores = {} # Stores average review scores for products
product_total_helpfulness = {} # Stores TOTAL helpfulness numerator for products
user_profile_names = {} # Stores mapping of UserId to ProfileName

def load_and_preprocess_data():
    """
    Loads the dataset and performs all necessary preprocessing steps
    as done in the Jupyter notebook.
    This function will be called once when the Flask app starts.
    """
    global df_final, pivot_df, preds_df, user_id_to_iloc_idx, product_avg_scores, product_total_helpfulness, user_profile_names

    print("Loading and preprocessing data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found. Please ensure the CSV file is in the correct directory.")
        return False

    # Store UserId to ProfileName mapping before dropping columns
    user_profile_names = df.set_index('UserId')['ProfileName'].to_dict()

    # Calculate average scores for all products
    product_avg_scores = df.groupby('ProductId')['Score'].mean().round(2).to_dict()
    # Calculate TOTAL helpfulness numerator for all products
    product_total_helpfulness = df.groupby('ProductId')['HelpfulnessNumerator'].sum().to_dict()

    # Dropping columns as per notebook, but keeping 'HelpfulnessNumerator' for total calculation
    df = df.drop(['Id', 'Time','HelpfulnessDenominator','Text','Summary'], axis = 1)
    # Note: 'ProfileName' is dropped from this df now, but its mapping is already stored in user_profile_names

    # Filter users who have given 50 or more ratings
    counts = df['UserId'].value_counts()
    df_final = df[df['UserId'].isin(counts[counts >= 50].index)]

    # Create the pivot table for Collaborative Filtering
    pivot_df = pd.pivot_table(df_final, index=['UserId'], columns = 'ProductId', values = "Score")
    pivot_df.fillna(0, inplace=True)

    # Create the mapping from UserId (string) to its integer positional index (iloc_idx)
    user_id_to_iloc_idx = {user_id: idx for idx, user_id in enumerate(pivot_df.index)}

    # Perform SVD
    pivot_sparse = csr_matrix(pivot_df.values)
    
    k_value = min(50, pivot_sparse.shape[0] - 1, pivot_sparse.shape[1] - 1)
    if k_value <= 0:
        print("Error: Not enough data for SVD. Check the number of users or products after filtering.")
        return False

    U, sigma, Vt = svds(pivot_sparse, k=k_value)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=pivot_df.columns)

    print("Data preprocessing complete.")
    return True

# --- Recommendation Functions ---

def recommend_popularity(num_recommendations=5):
    """
    Generates popularity-based recommendations.
    Returns a list of dictionaries with 'ProductId', 'AvgScore', and 'TotalHelpfulness',
    sorted by AvgScore in descending order.
    """
    if df_final is None or not product_avg_scores or not product_total_helpfulness:
        return []

    train_data_grouped = df_final.groupby('ProductId').agg({'UserId': 'count'}).reset_index()
    train_data_grouped.rename(columns={'UserId': 'score'},inplace=True)
    train_data_sort = train_data_grouped.sort_values(['score', 'ProductId'], ascending = [0,1]) 
    popularity_recommendations = train_data_sort.head(num_recommendations)

    recommended_products_with_details = []
    for product_id in popularity_recommendations['ProductId'].tolist():
        avg_score = product_avg_scores.get(product_id, 'N/A')
        total_helpfulness = product_total_helpfulness.get(product_id, 'N/A')
        recommended_products_with_details.append({
            'ProductId': product_id,
            'AvgScore': avg_score,
            'TotalHelpfulness': total_helpfulness
        })
    
    # Sort the final list by AvgScore in descending order
    # Handle 'N/A' by treating it as a low value for sorting purposes
    recommended_products_with_details.sort(key=lambda x: x['AvgScore'] if isinstance(x['AvgScore'], (int, float)) else -1, reverse=True)

    return recommended_products_with_details

def recommend_svd(user_id, num_recommendations=10):
    """
    Generates SVD-based collaborative filtering recommendations for a given user.
    Returns a list of dictionaries with 'ProductId', 'AvgScore', and 'TotalHelpfulness',
    sorted by AvgScore in descending order.
    """
    global user_id_to_iloc_idx 

    if pivot_df is None or preds_df is None or user_id_to_iloc_idx is None or not product_avg_scores or not product_total_helpfulness:
        return []

    try:
        if user_id not in user_id_to_iloc_idx:
            return f"User ID '{user_id}' not found in the dataset with sufficient ratings (min 50 ratings required)."

        user_iloc_idx = user_id_to_iloc_idx[user_id]

        sorted_user_ratings = pivot_df.loc[user_id].sort_values(ascending=False)
        sorted_user_predictions = preds_df.iloc[user_iloc_idx].sort_values(ascending=False)

        temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1)
        temp.index.name = 'Recommended Items'
        temp.columns = ['user_ratings', 'user_predictions']

        temp = temp.loc[temp.user_ratings == 0]
        temp = temp.sort_values('user_predictions', ascending=False)

        recommended_products_ids = temp.head(num_recommendations).index.tolist()
        
        recommended_products_with_details = []
        for product_id in recommended_products_ids:
            avg_score = product_avg_scores.get(product_id, 'N/A')
            total_helpfulness = product_total_helpfulness.get(product_id, 'N/A')
            recommended_products_with_details.append({
                'ProductId': product_id,
                'AvgScore': avg_score,
                'TotalHelpfulness': total_helpfulness
            })
        
        # Sort the final list by AvgScore in descending order
        # Handle 'N/A' by treating it as a low value for sorting purposes
        recommended_products_with_details.sort(key=lambda x: x['AvgScore'] if isinstance(x['AvgScore'], (int, float)) else -1, reverse=True)

        return recommended_products_with_details
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
        num_recommendations = 5 

    model_type = request.form.get('model_type')

    recommendations = []
    error_message = None
    profile_name = "N/A"

    if model_type == 'popularity':
        recommendations = recommend_popularity(num_recommendations)
        if not recommendations:
            error_message = "Could not generate popularity recommendations. Data might not be loaded or processed correctly."
    elif model_type == 'collaborative':
        if not user_id:
            error_message = "Please enter a User ID for Collaborative Filtering."
        else:
            profile_name = user_profile_names.get(user_id, "Unknown Profile")
            
            recommendations = recommend_svd(user_id, num_recommendations)
            if isinstance(recommendations, str): 
                error_message = recommendations
                recommendations = [] 
            elif not recommendations:
                error_message = f"No collaborative filtering recommendations found for User ID '{user_id}'. This user might not have enough ratings or there are no unrated items to recommend."
    else:
        error_message = "Please select a recommendation model type."

    return render_template('recommendations.html',
                           user_id=user_id if model_type == 'collaborative' else 'N/A (Popularity)',
                           profile_name=profile_name,
                           num_recommendations=num_recommendations,
                           model_type=model_type,
                           recommendations=recommendations,
                           error_message=error_message)

# --- App Initialization ---
if __name__ == '__main__':
    if load_and_preprocess_data():
        app.run(debug=True)
    else:
        print("Application could not start due to data loading/preprocessing errors.")
