// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.

import type { LessonData } from "../types";

export const lesson: LessonData = {
  id: "c-machine-learning-fundamentals",
  title: "Machine Learning Fundamentals",
  description: "Explore machine learning concepts, algorithms, and practical applications with hands-on Python programming.",
  sections: [
    {
      title: "Understanding Machine Learning",
      content: `Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn and improve automatically from experience without being explicitly programmed for every task.

**Key Concepts:**
- **Pattern Recognition**: Identifying regularities in data
- **Prediction**: Making informed guesses about future or unknown data
- **Generalization**: Applying learned patterns to new, unseen data
- **Feature Extraction**: Identifying relevant characteristics in data

**Why Machine Learning Matters:**
- **Automation**: Automates complex decision-making processes
- **Scalability**: Handles massive datasets that humans can't process
- **Personalization**: Creates customized experiences (Netflix recommendations, social media feeds)
- **Innovation**: Enables breakthrough technologies like autonomous vehicles and medical diagnosis`
    },
    {
      title: "Types of Machine Learning",
      content: `Machine learning approaches can be categorized into three main types:

**1. Supervised Learning**
- **Definition**: Learning with labeled training data (input-output pairs)
- **Examples**: Email spam detection, medical diagnosis, stock price prediction

**2. Unsupervised Learning**
- **Definition**: Finding hidden patterns in data without labeled examples
- **Examples**: Customer segmentation, data compression, anomaly detection

**3. Reinforcement Learning**
- **Definition**: Learning through interaction with an environment using rewards/penalties
- **Examples**: Game AI (AlphaGo), autonomous driving, trading algorithms`
    },
    {
      title: "Linear Regression: Making Predictions",
      content: `Linear regression finds the best line that fits through data points to make predictions.

\`\`\`python interactive
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Sample data: study hours vs exam scores
study_hours = np.array([[2], [3], [4], [5], [6], [7], [8], [9]])
exam_scores = np.array([65, 70, 75, 80, 85, 90, 93, 96])

# Split data
X_train, X_test, y_train, y_test = train_test_split(study_hours, exam_scores, test_size=0.3, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Test with new data
new_hours = np.array([[6.5]])
predicted_score = model.predict(new_hours)
print(f"Predicted score for 6.5 hours of study: {predicted_score[0]:.1f}")

# Model performance
score = model.score(X_test, y_test)
print(f"Model accuracy (R² score): {score:.3f}")
\`\`\``
    },
    {
      title: "Classification with Decision Trees",
      content: `Decision trees make predictions by asking yes/no questions about the data.

\`\`\`python interactive
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample data: features for predicting if student passes
# Features: [study_hours, sleep_hours, attendance_rate]
X = np.array([
    [2, 6, 0.6], [8, 8, 0.95], [4, 7, 0.8], [1, 5, 0.5],
    [7, 7, 0.9], [3, 6, 0.7], [9, 8, 0.98], [2, 9, 0.6]
])

# Target: 1 = pass, 0 = fail
y = np.array([0, 1, 1, 0, 1, 0, 1, 0])

# Train decision tree
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X, y)

# Predict for new student
new_student = np.array([[6, 7, 0.85]])
prediction = tree_model.predict(new_student)
probability = tree_model.predict_proba(new_student)

print(f"Will student pass? {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Probability of passing: {probability[0][1]:.2f}")

# Feature importance
features = ['study_hours', 'sleep_hours', 'attendance_rate']
for feature, importance in zip(features, tree_model.feature_importances_):
    print(f"{feature}: {importance:.3f}")
\`\`\``
    },
    {
      title: "K-Means Clustering: Finding Groups",
      content: `K-means clustering groups similar data points without using labels.

\`\`\`python interactive
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample student data: [study_hours, social_media_hours]
students = np.array([
    [8, 1], [7, 2], [9, 1], [2, 8], [1, 9], [3, 7],
    [5, 4], [6, 3], [4, 5], [8, 2], [2, 8], [7, 1]
])

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(students)

# Analyze clusters
for i in range(3):
    cluster_students = students[clusters == i]
    avg_study = np.mean(cluster_students[:, 0])
    avg_social = np.mean(cluster_students[:, 1])
    print(f"Cluster {i}: {len(cluster_students)} students")
    print(f"  Average study hours: {avg_study:.1f}")
    print(f"  Average social media hours: {avg_social:.1f}")
    
    # Characterize cluster
    if avg_study > 6:
        print("  → High achievers")
    elif avg_social > 6:
        print("  → Social media focused")
    else:
        print("  → Balanced students")
    print()

# Predict cluster for new student
new_student = np.array([[7, 2]])
cluster = kmeans.predict(new_student)
print(f"New student (7 study hrs, 2 social hrs) belongs to Cluster {cluster[0]}")
\`\`\``
    },
    {
      title: "Model Evaluation and Validation",
      content: `Proper evaluation ensures your model works well on new data.

\`\`\`python interactive
from sklearn.model_selection import cross_val_score

# Create larger dataset for evaluation
np.random.seed(42)
X_large = np.random.rand(100, 3) * 10  # 100 students, 3 features
y_large = (X_large[:, 0] + X_large[:, 1] * 0.5 - X_large[:, 2] * 0.3 > 5).astype(int)

# Compare different models using cross-validation
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Linear Regression': LinearRegression()
}

print("Cross-Validation Results:")
for name, model in models.items():
    if name == 'Linear Regression':
        # Convert to classification
        scores = []
        for train_idx, test_idx in [[range(80), range(80, 100)]]:
            model.fit(X_large[train_idx], y_large[train_idx])
            pred = (model.predict(X_large[test_idx]) > 0.5).astype(int)
            scores.append(accuracy_score(y_large[test_idx], pred))
    else:
        scores = cross_val_score(model, X_large, y_large, cv=5, scoring='accuracy')
    
    print(f"{name}: {np.mean(scores):.3f} (+/- {np.std(scores) * 2:.3f})")

print("\\nKey Evaluation Tips:")
print("- Higher accuracy = better performance")
print("- Cross-validation gives more reliable estimates")
print("- Watch for overfitting (perfect training, poor testing)")
\`\`\``
    },
    {
      title: "Real-World Applications and Next Steps",
      content: `Machine learning is transforming many industries:

**🏥 Healthcare**: Medical image analysis, drug discovery
**🚗 Transportation**: Self-driving cars, route optimization  
**💰 Finance**: Fraud detection, trading algorithms
**🛒 E-commerce**: Recommendation systems, price optimization
**🎮 Entertainment**: Game AI, content recommendations

**Career Preparation:**
- **Math Skills**: Statistics, algebra, basic calculus
- **Programming**: Python, R, SQL
- **Domain Knowledge**: Choose an industry to specialize in
- **Communication**: Explain technical concepts clearly

**Next Steps:**
1. **Build Projects**: Create ML projects for your portfolio
2. **Join Competitions**: Try Kaggle for hands-on experience
3. **Take Courses**: Coursera, edX for structured learning
4. **Stay Current**: Follow ML news and research
5. **Network**: Join AI/ML communities and meetups

**High School Action Plan:**
- Excel in math and computer science courses
- Learn Python and basic statistics
- Work on personal data science projects
- Consider summer internships or research opportunities

The field of machine learning offers excellent career prospects with strong technical skills and continuous learning!`
    }
  ]
}; 