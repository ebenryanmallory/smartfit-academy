// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.
import { LessonData } from "./lesson-1";

export const lesson3: LessonData = {
  id: 3,
  title: "Machine Learning Basics",
  description: "Explore the core concepts of machine learning and how it powers modern AI systems.",
  sections: [
    {
      title: "What is Machine Learning?",
      content: `Machine Learning (ML) is a branch of Artificial Intelligence that enables computers to learn from data and improve over time without being explicitly programmed.`
    },
    {
      title: "Types of Machine Learning",
      content: `1. **Supervised Learning**: The model learns from labeled data (e.g., classifying emails as spam or not).
2. **Unsupervised Learning**: The model finds patterns in unlabeled data (e.g., grouping customers by purchasing behavior).
3. **Reinforcement Learning**: The model learns by trial and error through rewards and penalties (e.g., training a robot to walk).`
    },
    {
      title: "Making a Simple Prediction (No Libraries)",
      content: `Let's start with the simplest form of prediction: using a basic formula without any libraries.

\`\`\`python
def predict(x):
    # A hardcoded linear model: y = 2x + 1
    return 2 * x + 1

print(predict(3))  # Output: 7
\`\`\`
`
    },
    {
      title: "Basic Data Manipulation with NumPy",
      content: `NumPy is a popular library for working with numerical data in Python. Let's see how to use it to calculate the mean of a dataset.

\`\`\`python
import numpy as np

data = [1, 2, 3, 4, 5]
mean = np.mean(data)
print(f"Mean: {mean}")
\`\`\`
`
    },
    {
      title: "A Simple Linear Regression Example",
      content: `Let's see how to fit a line to data using Python's scikit-learn library.

\`\`\`python
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4], [5]])  # Features
y = np.array([2, 4, 5, 4, 5])           # Labels

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make a prediction
prediction = model.predict(np.array([[6]]))
print(f"Prediction for 6: {prediction[0]:.2f}")
\`\`\`
`
    },
    {
      title: "Simple Classification with scikit-learn",
      content: `Let's classify data using a basic k-Nearest Neighbors (k-NN) classifier.

\`\`\`python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Example data
X = np.array([[0], [1], [2], [3]])    # Features
y = np.array([0, 0, 1, 1])            # Labels

# Create and train the classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

# Make a prediction
print(knn.predict([[1.5]]))  # Output: [0]
\`\`\`
`
    },
    {
      title: "Basic Clustering (Unsupervised Learning)",
      content: `Clustering is a way to group similar data points together. Let's use k-means clustering.

\`\`\`python
from sklearn.cluster import KMeans
import numpy as np

# Example data
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# Create and fit the model
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

print(f"Cluster centers: {kmeans.cluster_centers_}")
print(f"Labels: {kmeans.labels_}")
\`\`\`
`
    },
    {
      title: "Key Terms",
      content: `- **Feature**: An input variable used for prediction.
- **Label**: The output variable or value to predict.
- **Model**: The algorithm that learns from data to make predictions.
- **Training**: The process of teaching a model using data.`
    }
  ]
};
