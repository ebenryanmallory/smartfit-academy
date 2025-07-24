// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.

import type { LessonData } from "../types";

export const lesson: LessonData = {
  id: "c-machine-learning-fundamentals",
  title: "Machine Learning Fundamentals for Computer Science Students",
  description: "Comprehensive introduction to machine learning concepts, algorithms, and practical implementation for undergraduate computer science students.",
  sections: [
    {
      title: "Introduction to Machine Learning",
      content: `Machine Learning (ML) is a subset of Artificial Intelligence that enables computers to learn patterns from data and make predictions or decisions without being explicitly programmed for each specific task.

**Core Concepts:**
- **Data-Driven Learning**: Instead of writing rules, we provide examples
- **Pattern Recognition**: Identifying relationships and structures in data
- **Generalization**: Applying learned patterns to new, unseen data
- **Automation**: Reducing human intervention in decision-making processes

**Why Machine Learning Matters in Computer Science:**
- **Scalability**: Handle problems too complex for traditional programming
- **Adaptability**: Systems that improve automatically with more data
- **Innovation**: Enables breakthrough applications like speech recognition and autonomous vehicles
- **Career Relevance**: Essential skill for modern software development

**Real-World Impact:**
- Search engines ranking web pages
- Social media recommendation systems
- Email spam filtering
- Medical diagnosis assistance
- Financial fraud detection`
    },
    {
      title: "Types of Machine Learning",
      content: `Machine learning approaches are categorized based on the type of learning signal or feedback available:

**1. Supervised Learning**
- **Definition**: Learning with labeled training examples (input-output pairs)
- **Goal**: Map inputs to correct outputs for new data
- **Types**:
  - *Classification*: Predicting categories (spam/not spam, cat/dog)
  - *Regression*: Predicting continuous values (stock prices, temperature)
- **Examples**: Email classification, medical diagnosis, price prediction

**2. Unsupervised Learning**
- **Definition**: Finding hidden patterns in data without labeled examples
- **Goal**: Discover structure in data
- **Types**:
  - *Clustering*: Grouping similar data points
  - *Association*: Finding relationships between variables
  - *Dimensionality Reduction*: Simplifying data while preserving information
- **Examples**: Customer segmentation, data compression, anomaly detection

**3. Reinforcement Learning**
- **Definition**: Learning through interaction with an environment using rewards/penalties
- **Goal**: Maximize cumulative reward over time
- **Components**: Agent, environment, actions, rewards, states
- **Examples**: Game AI (chess, Go), robotics, autonomous driving

**Choosing the Right Approach:**
- Supervised: When you have labeled data and clear target outcomes
- Unsupervised: When exploring data structure or no clear target exists
- Reinforcement: When learning through trial-and-error in interactive environments`
    },
    {
      title: "Essential Data Science Libraries",
      content: `Before diving into machine learning algorithms, let's explore the foundational libraries that make ML accessible in Python:

\`\`\`python interactive
# Essential imports for machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# NumPy: Numerical computing foundation
print("NumPy - Numerical Computing:")
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Mean: {np.mean(data):.2f}")
print(f"Standard deviation: {np.std(data):.2f}")
print(f"Shape: {data.shape}")

# Working with 2D arrays (matrices)
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\\nMatrix shape: {matrix.shape}")
print(f"Matrix transpose:\\n{matrix.T}")

# Pandas: Data manipulation and analysis
print("\\nPandas - Data Manipulation:")
# Create a sample dataset
students_data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'math_score': [85, 78, 92, 88, 76],
    'science_score': [90, 82, 89, 91, 84],
    'study_hours': [5, 3, 7, 6, 4]
}

df = pd.DataFrame(students_data)
print(df)
print(f"\\nAverage math score: {df['math_score'].mean():.1f}")
print(f"Correlation between study hours and math score: {df['study_hours'].corr(df['math_score']):.3f}")

# Basic data exploration
print(f"\\nDataset info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\\n{df.dtypes}")
\`\`\`

**Key Libraries Overview:**
- **NumPy**: Fast numerical operations, array handling
- **Pandas**: Data manipulation, CSV/Excel reading, data cleaning
- **Matplotlib/Seaborn**: Data visualization and plotting
- **Scikit-learn**: Machine learning algorithms and tools
- **Jupyter**: Interactive development environment`
    },
    {
      title: "Linear Regression: Understanding Relationships",
      content: `Linear regression finds the best line that fits through data points to model relationships between variables.

\`\`\`python interactive
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create sample data: relationship between study hours and exam scores
np.random.seed(42)  # For reproducible results
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
# Linear relationship with some noise
exam_scores = 60 + 4 * study_hours.flatten() + np.random.normal(0, 3, 10)

print("Study Hours vs Exam Scores Dataset:")
for i in range(len(study_hours)):
    print(f"{study_hours[i][0]} hours -> {exam_scores[i]:.1f} score")

# Create and train the linear regression model
model = LinearRegression()
model.fit(study_hours, exam_scores)

# Make predictions
predictions = model.predict(study_hours)

# Model evaluation
mse = mean_squared_error(exam_scores, predictions)
r2 = r2_score(exam_scores, predictions)

print(f"\\nModel Performance:")
print(f"Slope (coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Equation: Score = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Hours")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.3f}")

# Predict for new values
new_hours = np.array([[5.5], [12]])
new_predictions = model.predict(new_hours)
print(f"\\nPredictions:")
print(f"5.5 hours of study -> {new_predictions[0]:.1f} score")
print(f"12 hours of study -> {new_predictions[1]:.1f} score")

# Understanding the model
print(f"\\nModel Interpretation:")
print(f"Each additional hour of study increases score by {model.coef_[0]:.2f} points")
print(f"A student who doesn't study gets {model.intercept_:.1f} points on average")
\`\`\`

**Key Concepts:**
- **Slope**: How much Y changes for each unit change in X
- **Intercept**: Y value when X = 0
- **R² Score**: Proportion of variance explained (0-1, higher is better)
- **Mean Squared Error**: Average squared difference between actual and predicted values`
    },
    {
      title: "Classification with Decision Trees",
      content: `Decision trees make predictions by asking a series of yes/no questions about the data features.

\`\`\`python interactive
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Create a more comprehensive dataset for student pass/fail prediction
np.random.seed(42)
n_students = 200

# Generate features
study_hours = np.random.normal(6, 2, n_students)
attendance = np.random.uniform(0.5, 1.0, n_students)
prev_gpa = np.random.normal(3.0, 0.5, n_students)
sleep_hours = np.random.normal(7, 1.5, n_students)

# Create realistic pass/fail outcomes based on features
pass_probability = (
    0.3 * (study_hours / 10) +
    0.4 * attendance +
    0.2 * (prev_gpa / 4.0) +
    0.1 * np.minimum(sleep_hours / 8, 1)
)

# Add some randomness
pass_fail = (pass_probability + np.random.normal(0, 0.1, n_students)) > 0.6

# Create feature matrix
X = np.column_stack([study_hours, attendance, prev_gpa, sleep_hours])
y = pass_fail.astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Dataset: {len(X)} students")
print(f"Training set: {len(X_train)} students")
print(f"Test set: {len(X_test)} students")
print(f"Pass rate: {np.mean(y):.1%}")

# Create and train decision tree
tree_model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
tree_model.fit(X_train, y_train)

# Make predictions
train_pred = tree_model.predict(X_train)
test_pred = tree_model.predict(X_test)

# Evaluate performance
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"\\nModel Performance:")
print(f"Training accuracy: {train_accuracy:.3f}")
print(f"Test accuracy: {test_accuracy:.3f}")

# Feature importance
feature_names = ['Study Hours', 'Attendance', 'Previous GPA', 'Sleep Hours']
importances = tree_model.feature_importances_

print(f"\\nFeature Importance:")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.3f}")

# Predict for sample students
sample_students = np.array([
    [8, 0.9, 3.5, 7],   # High performer
    [3, 0.6, 2.5, 5],   # Struggling student
    [6, 0.8, 3.0, 7]    # Average student
])

sample_predictions = tree_model.predict(sample_students)
sample_probabilities = tree_model.predict_proba(sample_students)

print(f"\\nSample Predictions:")
for i, (student, pred, prob) in enumerate(zip(sample_students, sample_predictions, sample_probabilities)):
    print(f"Student {i+1}: {feature_names[0]}={student[0]:.1f}, {feature_names[1]}={student[1]:.1f}, {feature_names[2]}={student[2]:.1f}, {feature_names[3]}={student[3]:.1f}")
    print(f"  Prediction: {'Pass' if pred else 'Fail'} (confidence: {prob[pred]:.2f})")
\`\`\`

**Decision Tree Advantages:**
- **Interpretable**: Easy to understand the decision process
- **No assumptions**: Works with any type of data distribution
- **Feature selection**: Automatically identifies important features
- **Non-linear**: Can capture complex relationships
    `},
    {
      title: "K-Means Clustering: Finding Hidden Groups",
      content: `K-means clustering automatically groups similar data points together without using labels.

\`\`\`python interactive
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Create customer data for an online store
np.random.seed(42)
n_customers = 300

# Generate customer features
age = np.random.normal(35, 12, n_customers)
annual_income = np.random.normal(50000, 20000, n_customers)
spending_score = np.random.normal(50, 25, n_customers)  # 1-100 scale

# Ensure reasonable bounds
age = np.clip(age, 18, 70)
annual_income = np.clip(annual_income, 20000, 120000)
spending_score = np.clip(spending_score, 1, 100)

# Create feature matrix
X = np.column_stack([age, annual_income, spending_score])

# Standardize features (important for k-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Customer Segmentation Analysis")
print(f"Dataset: {n_customers} customers")
print(f"Features: Age, Annual Income, Spending Score")

# Determine optimal number of clusters using elbow method
inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Apply k-means with optimal k (let's use 4 clusters)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Analyze clusters
print(f"\\nCluster Analysis (k={optimal_k}):")
for cluster_id in range(optimal_k):
    cluster_mask = cluster_labels == cluster_id
    cluster_data = X[cluster_mask]
    
    print(f"\\nCluster {cluster_id}: {np.sum(cluster_mask)} customers")
    print(f"  Average age: {np.mean(cluster_data[:, 0]):.1f} years")
    print(f"  Average income: \${np.mean(cluster_data[:, 1]):,.0f}")
    print(f"  Average spending score: {np.mean(cluster_data[:, 2]):.1f}")
    
    # Characterize cluster
    avg_age = np.mean(cluster_data[:, 0])
    avg_income = np.mean(cluster_data[:, 1])
    avg_spending = np.mean(cluster_data[:, 2])
    
    if avg_income > 60000 and avg_spending > 60:
        cluster_type = "High Value Customers"
    elif avg_income < 40000 and avg_spending < 40:
        cluster_type = "Budget Conscious"
    elif avg_spending > 70:
        cluster_type = "High Spenders"
    elif avg_age < 30:
        cluster_type = "Young Adults"
    else:
        cluster_type = "Moderate Customers"
    
    print(f"  Profile: {cluster_type}")

# Cluster centers in original scale
cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)

print(f"\\nCluster Centers (Original Scale):")
for i, center in enumerate(cluster_centers_original):
    print(f"Cluster {i}: Age={center[0]:.1f}, Income=\${center[1]:,.0f}, Spending={center[2]:.1f}")

# Predict cluster for new customers
new_customers = np.array([
    [25, 40000, 80],   # Young high spender
    [45, 80000, 30],   # Middle-aged conservative spender
    [35, 60000, 65]    # Balanced customer
])

new_customers_scaled = scaler.transform(new_customers)
new_clusters = kmeans.predict(new_customers_scaled)

print(f"\\nNew Customer Predictions:")
for i, (customer, cluster) in enumerate(zip(new_customers, new_clusters)):
    print(f"Customer {i+1}: Age={customer[0]}, Income=\${customer[1]:,}, Spending={customer[2]} -> Cluster {cluster}")
\`\`\`

**K-Means Key Concepts:**
- **Centroids**: Center points of each cluster
- **Inertia**: Sum of squared distances from points to their cluster center
- **Elbow Method**: Finding optimal number of clusters
- **Standardization**: Scaling features to prevent bias from different units
    `},
    {
      title: "Model Evaluation and Validation",
      content: `Proper evaluation ensures your model works well on new, unseen data and helps prevent overfitting.

\`\`\`python interactive
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

# Create a binary classification dataset
np.random.seed(42)
n_samples = 1000

# Features: student academic data
study_time = np.random.exponential(5, n_samples)  # Hours per week
assignment_scores = np.random.normal(75, 15, n_samples)
attendance_rate = np.random.beta(8, 2, n_samples)  # Skewed toward high attendance
previous_course_grade = np.random.normal(3.0, 0.8, n_samples)  # GPA scale

# Create realistic pass/fail outcomes
pass_probability = (
    0.2 * np.minimum(study_time / 10, 1) +
    0.3 * (assignment_scores / 100) +
    0.3 * attendance_rate +
    0.2 * (previous_course_grade / 4.0)
)

# Add noise and create binary outcome
final_grade = pass_probability + np.random.normal(0, 0.1, n_samples)
passed = (final_grade > 0.6).astype(int)

# Create feature matrix
X = np.column_stack([study_time, assignment_scores, attendance_rate, previous_course_grade])
y = passed

print("Model Evaluation Demonstration")
print(f"Dataset: {n_samples} students")
print(f"Pass rate: {np.mean(y):.1%}")

# Split data with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train multiple models for comparison
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

print("\\nModel Comparison:")
for name, model in models.items():
    # Cross-validation for robust evaluation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Train on full training set and evaluate on test set
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\\n{name}:")
    print(f"  Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print(f"  Training accuracy: {train_score:.3f}")
    print(f"  Test accuracy: {test_score:.3f}")
    
    # Check for overfitting
    if train_score - test_score > 0.05:
        print(f"  ⚠️  Possible overfitting detected!")

# Detailed evaluation of best model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Multiple evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\\nDetailed Random Forest Evaluation:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f} (of predicted passes, how many actually passed)")
print(f"Recall: {recall:.3f} (of actual passes, how many were predicted)")
print(f"F1-score: {f1:.3f} (harmonic mean of precision and recall)")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\\nConfusion Matrix:")
print(f"                Predicted")
print(f"Actual    Fail   Pass")
print(f"Fail      {cm[0,0]:4d}   {cm[0,1]:4d}")
print(f"Pass      {cm[1,0]:4d}   {cm[1,1]:4d}")

# Feature importance
feature_names = ['Study Time', 'Assignment Scores', 'Attendance Rate', 'Previous Grade']
importances = rf_model.feature_importances_

print(f"\\nFeature Importance:")
sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.3f}")

# Validation curve to check model complexity
param_range = [10, 50, 100, 200, 300]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(random_state=42), X_train, y_train,
    param_name='n_estimators', param_range=param_range,
    cv=3, scoring='accuracy'
)

print(f"\\nValidation Curve (n_estimators):")
for n_est, train_mean, val_mean in zip(param_range, train_scores.mean(axis=1), val_scores.mean(axis=1)):
    print(f"n_estimators={n_est:3d}: Train={train_mean:.3f}, Validation={val_mean:.3f}")
\`\`\`

**Key Evaluation Concepts:**
- **Cross-Validation**: Multiple train/validation splits for robust evaluation
- **Overfitting**: Model performs well on training data but poorly on new data
- **Underfitting**: Model is too simple to capture underlying patterns
- **Precision vs Recall**: Trade-off between false positives and false negatives`
    },
    {
      title: "Practical Project: Movie Recommendation System",
      content: `Let's build a complete machine learning project that demonstrates the full pipeline from data preparation to model deployment.

\`\`\`python interactive
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Create a movie dataset
np.random.seed(42)
n_movies = 500
n_users = 1000

# Generate movie data
genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
movie_data = {
    'movie_id': range(1, n_movies + 1),
    'title': [f"Movie_{i}" for i in range(1, n_movies + 1)],
    'genre': np.random.choice(genres, n_movies),
    'year': np.random.randint(1980, 2024, n_movies),
    'duration': np.random.normal(110, 20, n_movies),  # minutes
    'budget': np.random.exponential(50, n_movies),     # millions
    'avg_rating': np.random.normal(6.5, 1.5, n_movies)
}

# Ensure realistic constraints
movie_data['duration'] = np.clip(movie_data['duration'], 80, 180)
movie_data['avg_rating'] = np.clip(movie_data['avg_rating'], 1, 10)

movies_df = pd.DataFrame(movie_data)

# Generate user rating data
user_ratings = []
for user_id in range(1, min(101, n_users + 1)):  # Limit for demonstration
    # Each user rates 10-50 movies
    n_ratings = np.random.randint(10, 51)
    rated_movies = np.random.choice(movie_data['movie_id'], n_ratings, replace=False)
    
    for movie_id in rated_movies:
        movie_idx = movie_id - 1
        base_rating = movies_df.iloc[movie_idx]['avg_rating']
        # Add user preference noise
        user_rating = base_rating + np.random.normal(0, 1.5)
        user_rating = np.clip(user_rating, 1, 10)
        
        user_ratings.append({
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': user_rating
        })

ratings_df = pd.DataFrame(user_ratings)

print("Movie Recommendation System")
print(f"Movies: {len(movies_df)}")
print(f"Users: {ratings_df['user_id'].nunique()}")
print(f"Ratings: {len(ratings_df)}")
print(f"\\nSample movie data:")
print(movies_df.head())

print(f"\\nSample ratings:")
print(ratings_df.head())

# Content-based filtering using movie features
def content_based_recommendations(movie_id, movies_df, top_n=5):
    """Recommend movies similar to a given movie based on content features."""
    
    # Create feature matrix
    # One-hot encode genres
    genre_dummies = pd.get_dummies(movies_df['genre'], prefix='genre')
    
    # Normalize numerical features
    features = movies_df[['year', 'duration', 'budget', 'avg_rating']].copy()
    features = (features - features.mean()) / features.std()
    
    # Combine features
    feature_matrix = pd.concat([features, genre_dummies], axis=1)
    
    # Calculate similarity
    similarity_matrix = cosine_similarity(feature_matrix)
    
    # Get recommendations
    movie_idx = movie_id - 1
    similarities = similarity_matrix[movie_idx]
    similar_indices = similarities.argsort()[::-1][1:top_n+1]  # Exclude self
    
    recommendations = movies_df.iloc[similar_indices][['title', 'genre', 'year', 'avg_rating']]
    recommendations['similarity'] = similarities[similar_indices]
    
    return recommendations

# Collaborative filtering using matrix factorization approach
def collaborative_filtering_model(ratings_df, movies_df):
    """Build a collaborative filtering model using movie and user features."""
    
    # Create user-movie matrix
    user_movie_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating')
    
    # Prepare training data
    train_data = []
    for _, row in ratings_df.iterrows():
        user_id, movie_id, rating = row['user_id'], row['movie_id'], row['rating']
        
        # User features (simplified)
        user_avg_rating = ratings_df[ratings_df['user_id'] == user_id]['rating'].mean()
        user_rating_count = len(ratings_df[ratings_df['user_id'] == user_id])
        
        # Movie features
        movie_info = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
        
        # Combine features
        features = [
            user_avg_rating,
            user_rating_count,
            movie_info['year'],
            movie_info['duration'],
            movie_info['budget'],
            movie_info['avg_rating'],
            1 if movie_info['genre'] == 'Action' else 0,
            1 if movie_info['genre'] == 'Comedy' else 0,
            1 if movie_info['genre'] == 'Drama' else 0,
        ]
        
        train_data.append(features + [rating])
    
    # Convert to arrays
    train_data = np.array(train_data)
    X = train_data[:, :-1]
    y = train_data[:, -1]
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\\nCollaborative Filtering Model Performance:")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    
    return model

# Demonstrate content-based recommendations
print(f"\\nContent-Based Recommendations:")
sample_movie_id = 1
sample_movie = movies_df[movies_df['movie_id'] == sample_movie_id].iloc[0]
print(f"Base movie: {sample_movie['title']} ({sample_movie['genre']}, {sample_movie['year']})")

recommendations = content_based_recommendations(sample_movie_id, movies_df)
print(f"\\nSimilar movies:")
for _, movie in recommendations.iterrows():
    print(f"  {movie['title']} ({movie['genre']}, {movie['year']}) - Similarity: {movie['similarity']:.3f}")

# Build collaborative filtering model
cf_model = collaborative_filtering_model(ratings_df, movies_df)

# Feature importance for collaborative filtering
feature_names = [
    'User Avg Rating', 'User Rating Count', 'Movie Year', 'Movie Duration',
    'Movie Budget', 'Movie Avg Rating', 'Genre: Action', 'Genre: Comedy', 'Genre: Drama'
]

print(f"\\nFeature Importance in Collaborative Filtering:")
for name, importance in zip(feature_names, cf_model.feature_importances_):
    print(f"{name}: {importance:.3f}")

# Hybrid recommendation function
def hybrid_recommendations(user_id, movie_id, movies_df, ratings_df, cf_model, alpha=0.6):
    """Combine content-based and collaborative filtering recommendations."""
    
    # Content-based score
    content_recs = content_based_recommendations(movie_id, movies_df, top_n=10)
    
    # Collaborative filtering score (simplified)
    user_avg = ratings_df[ratings_df['user_id'] == user_id]['rating'].mean()
    movie_avg = movies_df[movies_df['movie_id'] == movie_id]['avg_rating'].iloc[0]
    
    # Weighted combination
    hybrid_score = alpha * content_recs['similarity'].mean() + (1 - alpha) * (user_avg + movie_avg) / 2
    
    return hybrid_score

print(f"\\nHybrid Recommendation System:")
print(f"Combines content-based filtering (movie similarity) with collaborative filtering (user preferences)")
print(f"Alpha = 0.6 means 60% weight on content, 40% on collaboration")

# Recommendation system evaluation
print(f"\\nRecommendation System Summary:")
print(f"✅ Content-based: Good for new movies, explains recommendations")
print(f"✅ Collaborative: Good for personalization, finds unexpected connections")
print(f"✅ Hybrid: Combines strengths, more robust recommendations")
\`\`\`

**Project Components Covered:**
- **Data Generation**: Creating realistic synthetic datasets
- **Feature Engineering**: Transforming raw data into ML-ready features
- **Content-Based Filtering**: Using item features for recommendations
- **Collaborative Filtering**: Using user-item interactions
- **Model Evaluation**: Measuring recommendation quality
- **Hybrid Approaches**: Combining multiple techniques`
    },
    {
      title: "Career Paths and Next Steps in Machine Learning",
      content: `Machine learning offers diverse career opportunities for computer science graduates:

**🎯 Career Roles:**

**Data Scientist**
- Analyze data to extract business insights
- Build predictive models for decision-making
- Skills: Statistics, ML algorithms, domain expertise
- Tools: Python/R, SQL, Tableau, Jupyter

**Machine Learning Engineer**
- Deploy ML models into production systems
- Focus on scalability and performance
- Skills: Software engineering, MLOps, cloud platforms
- Tools: Docker, Kubernetes, AWS/GCP, TensorFlow/PyTorch

**Research Scientist**
- Develop new ML algorithms and techniques
- Publish papers and advance the field
- Skills: Strong math background, research methodology
- Environment: Tech companies, universities, research labs

**Product Manager (AI/ML)**
- Guide development of ML-powered products
- Bridge technical teams and business needs
- Skills: Technical understanding, business acumen
- Focus: Product strategy, user experience

**🛠 Essential Skills to Develop:**

**Programming**
- **Python**: Primary ML language, rich ecosystem
- **SQL**: Database queries and data manipulation
- **Git**: Version control for collaborative development
- **Cloud Platforms**: AWS, Google Cloud, Azure

**Mathematics and Statistics**
- **Linear Algebra**: Vectors, matrices, eigenvalues
- **Calculus**: Optimization and gradients
- **Statistics**: Probability, hypothesis testing, Bayesian methods
- **Discrete Math**: Algorithms and complexity analysis

**Machine Learning Specializations**
- **Computer Vision**: Image and video analysis
- **Natural Language Processing**: Text and language understanding
- **Recommender Systems**: Personalization and filtering
- **Time Series**: Forecasting and temporal data
- **Reinforcement Learning**: Decision-making and control

**📚 Learning Resources and Next Steps:**

**Online Courses**
- Andrew Ng's Machine Learning Course (Coursera)
- Fast.ai Practical Deep Learning
- CS229 Stanford Machine Learning (YouTube)
- MIT OpenCourseWare 6.034 Artificial Intelligence

**Books for Deeper Understanding**
- "Hands-On Machine Learning" by Aurélien Géron
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani, Friedman
- "Introduction to Statistical Learning" by James, Witten, Hastie, Tibshirani

**Practical Projects to Build**
1. **Predictive Analytics**: Stock price prediction, sales forecasting
2. **Classification**: Image recognition, sentiment analysis
3. **Recommendation Engine**: Movie/book recommender
4. **Natural Language Processing**: Chatbot, text summarization
5. **Computer Vision**: Object detection, facial recognition

**Portfolio Development**
- **GitHub**: Showcase code and projects
- **Kaggle**: Participate in competitions
- **Medium/Blog**: Write about your projects and learnings
- **LinkedIn**: Professional networking and visibility

**Industry Applications**
- **Healthcare**: Medical imaging, drug discovery, personalized medicine
- **Finance**: Fraud detection, algorithmic trading, risk assessment
- **Technology**: Search engines, recommendation systems, autonomous vehicles
- **Retail**: Demand forecasting, price optimization, customer segmentation
- **Entertainment**: Content recommendation, game AI, creative tools

**Graduate School Considerations**
- **MS in Computer Science**: Broader CS background with ML focus
- **MS in Data Science**: Interdisciplinary program combining CS, stats, domain knowledge
- **MS in AI/ML**: Specialized programs focusing on artificial intelligence
- **PhD**: For research careers in academia or industrial research labs

**🚀 Getting Started Checklist:**

**Immediate (Next 1-3 months)**
- [ ] Complete a comprehensive ML course
- [ ] Learn pandas, numpy, scikit-learn thoroughly
- [ ] Build 2-3 small projects and put them on GitHub
- [ ] Start following ML researchers and practitioners on social media

**Short-term (3-6 months)**
- [ ] Participate in a Kaggle competition
- [ ] Learn deep learning with TensorFlow or PyTorch
- [ ] Choose a specialization area and dive deeper
- [ ] Attend virtual ML conferences or meetups

**Medium-term (6-12 months)**
- [ ] Build a substantial portfolio project
- [ ] Contribute to open-source ML projects
- [ ] Apply for internships or entry-level positions
- [ ] Consider graduate school applications if interested

The field of machine learning is rapidly evolving, offering exciting opportunities for those willing to continuously learn and adapt. Start with strong fundamentals, practice regularly, and build projects that demonstrate your skills to potential employers or graduate programs.

Remember: Every expert was once a beginner. The key is consistent practice and staying curious about how things work!`
    }
  ]
};
