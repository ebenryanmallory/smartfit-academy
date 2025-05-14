// Lesson 4: Introduction to Data Science with Python
// This lesson follows the conventions and structure of lesson-1.ts

import { LessonData } from "./lesson-1";

export const lesson4: LessonData = {
  id: 4,
  title: "Introduction to Data Science with Python",
  description: "Explore the basics of data science, including data manipulation, analysis, and visualization using Python.",
  sections: [
    {
      title: "What is Data Science?",
      content: `Data Science is the field of extracting insights and knowledge from data using scientific methods, processes, and systems. It combines programming, statistics, and domain expertise to analyze and interpret complex data. Data science is important because it helps organizations make data-driven decisions, uncover patterns, and predict future trends.`
    },
    {
      title: "Core Python Libraries for Data Science",
      content: `Python is the most popular language for data science due to its simplicity and powerful libraries:

1. **NumPy**: Efficient numerical computations and array operations.
2. **Pandas**: Data manipulation and analysis with labeled data structures.
3. **Matplotlib & Seaborn**: Data visualization tools.
4. **scikit-learn**: Machine learning algorithms and utilities.

These libraries form the foundation of most data science workflows.`
    },
    {
      title: "Working with DataFrames (Pandas)",
      content: `A DataFrame is a table-like data structure from the pandas library. It's essential for organizing and analyzing data.

\`\`\`python
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [85, 92, 88]
}
df = pd.DataFrame(data)
print(df)
\`\`\`

This code creates a simple DataFrame. DataFrames allow you to filter, aggregate, and transform data easily.`
    },
    {
      title: "Reading and Inspecting Data",
      content: `Real-world data often comes in CSV files. Pandas makes loading and inspecting data simple:

\`\`\`python
import pandas as pd

df = pd.read_csv('data.csv')
print(df.head())  # Show the first 5 rows
print(df.info())  # Summary of columns and data types
\`\`\`

Inspecting data helps you understand its structure and spot issues early.`
    },
    {
      title: "Data Cleaning: Handling Missing Values",
      content: `Data is rarely perfect. Handling missing values is a key part of data cleaning:

\`\`\`python
# Drop rows with any missing values
df_clean = df.dropna()

# Fill missing values with a default
df_filled = df.fillna(0)
\`\`\`

Cleaning data ensures your analysis is accurate and reliable.`
    },
    {
      title: "Descriptive Statistics",
      content: `Descriptive statistics summarize and describe data features:

\`\`\`python
print(df['Score'].mean())    # Average score
print(df['Age'].median())    # Median age
print(df.describe())         # Summary statistics for all columns
\`\`\`

These functions help you quickly understand key properties of your dataset.`
    },
    {
      title: "Basic Data Visualization",
      content: `Visualization helps you see patterns and trends. Here's a histogram and a scatter plot:

# Histogram
\`\`\`python
import matplotlib.pyplot as plt

df['Score'].hist()
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title('Distribution of Scores')
plt.show()
\`\`\`

# Scatter Plot
\`\`\`python
plt.scatter(df['Age'], df['Score'])
plt.xlabel('Age')
plt.ylabel('Score')
plt.title('Score vs Age')
plt.show()
\`\`\`

Visualizations make data insights more accessible and actionable.`
    },
    {
      title: "Introduction to Machine Learning (scikit-learn)",
      content: `Machine learning lets computers learn from data. scikit-learn makes it easy to train simple models:

\`\`\`python
from sklearn.linear_model import LinearRegression
import numpy as np

X = df[['Age']].values  # Feature(s)
y = df['Score'].values  # Target

model = LinearRegression()
model.fit(X, y)

print('Predicted score for age 28:', model.predict(np.array([[28]])))
\`\`\`

This fits a linear model to predict Score based on Age. Even this simple model can provide valuable insights!`
    },
    {
      title: "Summary and Next Steps",
      content: `In this lesson, you learned about the core Python libraries for data science, how to load and clean data, compute statistics, visualize results, and run a simple machine learning model. These are the building blocks for deeper data science exploration!`
    }
  ]
};
