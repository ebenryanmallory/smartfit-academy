// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.

import type { LessonData } from "../types";

export const lesson4: LessonData = {
  id: 4,
  title: "Data Science Fundamentals with Python",
  description: "Comprehensive introduction to data science concepts, Python libraries, statistical analysis, and machine learning for undergraduate computer science students.",
  sections: [
    {
      title: "Introduction to Data Science",
      content: `Data Science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data.

**What Makes Data Science Unique?**
- **Interdisciplinary Approach**: Combines computer science, statistics, mathematics, and domain expertise
- **Data-Driven Decision Making**: Uses evidence rather than intuition to solve problems
- **Scalable Solutions**: Works with datasets ranging from hundreds to billions of records
- **Practical Impact**: Directly influences business decisions, scientific discoveries, and social policies

**The Data Science Process:**
1. **Problem Definition**: Understanding what question needs to be answered
2. **Data Collection**: Gathering relevant data from various sources
3. **Data Cleaning**: Preparing data for analysis by handling errors and inconsistencies
4. **Exploratory Data Analysis**: Understanding patterns and relationships in the data
5. **Modeling**: Building statistical or machine learning models
6. **Evaluation**: Assessing model performance and validity
7. **Communication**: Presenting findings to stakeholders

**Real-World Applications:**
- **Healthcare**: Predicting disease outbreaks, personalized medicine, drug discovery
- **Finance**: Credit scoring, fraud detection, algorithmic trading
- **Technology**: Recommendation systems, search engines, autonomous vehicles
- **Marketing**: Customer segmentation, A/B testing, targeted advertising
- **Sports**: Player performance analysis, game strategy optimization`
    },
    {
      title: "Python Data Science Ecosystem",
      content: `Python has become the leading language for data science due to its simplicity, readability, and extensive library ecosystem.

**Core Libraries Overview:**

**NumPy (Numerical Python)**
- Foundation for scientific computing in Python
- Provides N-dimensional array objects with vectorized operations
- Mathematical functions for linear algebra, Fourier transforms, and random number generation
- Memory efficient and fast operations on large datasets

**Pandas (Python Data Analysis Library)**
- High-level data manipulation and analysis tool
- Provides DataFrame and Series data structures
- Excel-like operations for data cleaning, transformation, and analysis
- Excellent for handling structured data (CSV, SQL, Excel files)

**Matplotlib**
- Comprehensive library for creating static, animated, and interactive visualizations
- Fine-grained control over plot appearance and customization
- Foundation for many other visualization libraries

**Seaborn**
- Statistical data visualization library built on matplotlib
- Beautiful default styles and color palettes
- High-level interface for drawing attractive statistical graphics

**Scikit-learn**
- Simple and efficient tools for machine learning and data mining
- Consistent API across different algorithms
- Includes classification, regression, clustering, and dimensionality reduction tools

\`\`\`python interactive
# Setting up the data science environment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Display library versions
print("Data Science Environment Setup:")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")

# Set visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("\\nEnvironment successfully configured!")
print("Ready for data science exploration!")
\`\`\`

**Why Python for Data Science?**
- **Readable Syntax**: Easy to learn and understand
- **Large Community**: Extensive documentation and community support
- **Integration**: Works well with databases, web frameworks, and deployment tools
- **Performance**: Can be optimized with NumPy, Cython, or integration with C/C++
- **Versatility**: Suitable for the entire data science pipeline from collection to deployment`
    },
    {
      title: "Working with DataFrames and Data Structures",
      content: `Pandas DataFrames are the cornerstone of data analysis in Python. Think of them as enhanced spreadsheets with powerful programming capabilities.

\`\`\`python interactive
import pandas as pd
import numpy as np

# Creating DataFrames from different sources
print("Creating DataFrames:")
print("="*30)

# From dictionary
student_data = {
    'student_id': [1, 2, 3, 4, 5],
    'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Ross', 'Eve Wilson'],
    'major': ['Computer Science', 'Mathematics', 'Physics', 'Computer Science', 'Statistics'],
    'gpa': [3.8, 3.6, 3.9, 3.7, 3.5],
    'credits': [90, 75, 95, 88, 82],
    'graduation_year': [2024, 2025, 2024, 2024, 2025]
}

df = pd.DataFrame(student_data)
print("Student DataFrame:")
print(df)

# DataFrame properties and methods
print(f"\\nDataFrame shape: {df.shape}")  # (rows, columns)
print(f"Column names: {list(df.columns)}")
print(f"Data types:\\n{df.dtypes}")
print(f"\\nDataFrame info:")
print(df.info())

# Basic DataFrame operations
print("\\nBasic Operations:")
print(f"First 3 rows:\\n{df.head(3)}")
print(f"\\nLast 2 rows:\\n{df.tail(2)}")
print(f"\\nRandom sample of 2 rows:\\n{df.sample(2)}")

# Selecting data
print("\\nSelecting Data:")
print(f"Names column:\\n{df['name']}")
print(f"\\nMultiple columns:\\n{df[['name', 'major', 'gpa']]}")
print(f"\\nFirst 3 students' names and GPAs:\\n{df[['name', 'gpa']].head(3)}")
\`\`\`

**Key DataFrame Concepts:**
- **Index**: Row labels (automatically created or custom)
- **Columns**: Column labels with associated data types
- **Values**: The actual data stored in a 2D array
- **Shape**: Dimensions as (rows, columns)
- **dtypes**: Data types for each column`
    },
    {
      title: "Data Loading and Initial Exploration",
      content: `Real-world data comes from various sources. Pandas provides robust tools for loading and initially exploring datasets.

\`\`\`python interactive
# Simulating real-world data loading
# In practice, you'd use: df = pd.read_csv('filename.csv')

# Create a more realistic dataset
np.random.seed(42)
n_students = 200

# Generate synthetic student performance data
performance_data = {
    'student_id': range(1, n_students + 1),
    'age': np.random.normal(20, 2, n_students).round().astype(int),
    'study_hours_per_week': np.random.normal(15, 5, n_students).round(1),
    'previous_math_score': np.random.normal(75, 12, n_students).round(),
    'attendance_rate': np.random.beta(8, 2, n_students).round(3),
    'final_exam_score': np.random.normal(78, 15, n_students).round(),
    'course_grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], n_students, p=[0.2, 0.3, 0.3, 0.15, 0.05])
}

# Add some realistic constraints
performance_data['age'] = np.clip(performance_data['age'], 17, 25)
performance_data['study_hours_per_week'] = np.clip(performance_data['study_hours_per_week'], 0, 40)
performance_data['previous_math_score'] = np.clip(performance_data['previous_math_score'], 0, 100)
performance_data['final_exam_score'] = np.clip(performance_data['final_exam_score'], 0, 100)

df_performance = pd.DataFrame(performance_data)

print("Student Performance Dataset:")
print("="*40)

# Initial data exploration
print(f"Dataset shape: {df_performance.shape}")
print(f"\\nFirst 5 rows:")
print(df_performance.head())

print(f"\\nDataset information:")
print(df_performance.info())

print(f"\\nBasic statistics:")
print(df_performance.describe())

# Check for missing values
print(f"\\nMissing values per column:")
print(df_performance.isnull().sum())

# Unique values in categorical columns
print(f"\\nUnique course grades: {df_performance['course_grade'].unique()}")
print(f"Grade distribution:\\n{df_performance['course_grade'].value_counts()}")

# Data quality checks
print(f"\\nData Quality Checks:")
print(f"Age range: {df_performance['age'].min()} - {df_performance['age'].max()}")
print(f"Study hours range: {df_performance['study_hours_per_week'].min()} - {df_performance['study_hours_per_week'].max()}")
print(f"Attendance rate range: {df_performance['attendance_rate'].min()} - {df_performance['attendance_rate'].max()}")
\`\`\`

**Data Exploration Best Practices:**
1. **Always start with .info() and .describe()**: Get overview of data structure and basic statistics
2. **Check for missing values**: Use .isnull().sum() to identify data gaps
3. **Examine data types**: Ensure numeric data isn't stored as strings
4. **Look for outliers**: Use .describe() percentiles to spot unusual values
5. **Understand categorical variables**: Use .value_counts() for categorical data distribution`
    },
    {
      title: "Data Cleaning and Preprocessing",
      content: `Real-world data is messy and requires cleaning before analysis. This critical step ensures accurate and reliable results.

\`\`\`python interactive
# Introduce some data quality issues to demonstrate cleaning
import pandas as pd
import numpy as np

# Create messy data similar to what you'd encounter in practice
messy_data = df_performance.copy()

# Introduce missing values randomly
np.random.seed(123)
missing_indices = np.random.choice(messy_data.index, size=20, replace=False)
messy_data.loc[missing_indices[:10], 'study_hours_per_week'] = np.nan
messy_data.loc[missing_indices[10:], 'attendance_rate'] = np.nan

# Introduce outliers
outlier_indices = np.random.choice(messy_data.index, size=5, replace=False)
messy_data.loc[outlier_indices, 'study_hours_per_week'] = np.random.uniform(60, 100, 5)

# Introduce data type issues
messy_data['final_exam_score'] = messy_data['final_exam_score'].astype(str)

print("Data Cleaning Process:")
print("="*30)

# 1. Identify data quality issues
print("1. Initial Data Quality Assessment:")
print(f"Missing values:\\n{messy_data.isnull().sum()}")
print(f"\\nData types:\\n{messy_data.dtypes}")

# 2. Handle missing values
print("\\n2. Handling Missing Values:")

# Strategy 1: Remove rows with missing values
print(f"Original dataset size: {len(messy_data)}")
clean_dropna = messy_data.dropna()
print(f"After dropping missing values: {len(clean_dropna)}")

# Strategy 2: Fill missing values
clean_filled = messy_data.copy()

# Fill study hours with median (robust to outliers)
median_study_hours = clean_filled['study_hours_per_week'].median()
clean_filled['study_hours_per_week'].fillna(median_study_hours, inplace=True)

# Fill attendance rate with mean (reasonable for this variable)
mean_attendance = clean_filled['attendance_rate'].mean()
clean_filled['attendance_rate'].fillna(mean_attendance, inplace=True)

print(f"Missing values after filling:\\n{clean_filled.isnull().sum()}")

# 3. Fix data types
print("\\n3. Fixing Data Types:")
clean_filled['final_exam_score'] = pd.to_numeric(clean_filled['final_exam_score'], errors='coerce')
print(f"Updated data types:\\n{clean_filled.dtypes}")

# 4. Handle outliers
print("\\n4. Handling Outliers:")
Q1 = clean_filled['study_hours_per_week'].quantile(0.25)
Q3 = clean_filled['study_hours_per_week'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Study hours IQR bounds: [{lower_bound:.1f}, {upper_bound:.1f}]")
outliers = clean_filled[(clean_filled['study_hours_per_week'] < lower_bound) | 
                       (clean_filled['study_hours_per_week'] > upper_bound)]
print(f"Number of outliers detected: {len(outliers)}")

# Cap outliers instead of removing them
clean_filled['study_hours_per_week'] = clean_filled['study_hours_per_week'].clip(lower_bound, upper_bound)

# 5. Create derived features
print("\\n5. Feature Engineering:")
# Create performance categories
clean_filled['performance_category'] = pd.cut(clean_filled['final_exam_score'], 
                                            bins=[0, 60, 70, 80, 90, 100], 
                                            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])

# Create study intensity categories
clean_filled['study_intensity'] = pd.cut(clean_filled['study_hours_per_week'], 
                                        bins=[0, 10, 20, 30, 100], 
                                        labels=['Low', 'Moderate', 'High', 'Very High'])

print(f"Performance categories:\\n{clean_filled['performance_category'].value_counts()}")
print(f"\\nStudy intensity distribution:\\n{clean_filled['study_intensity'].value_counts()}")

print("\\nData cleaning completed successfully!")
df_clean = clean_filled.copy()  # Use this for future analysis
\`\`\`

**Data Cleaning Best Practices:**
- **Document your decisions**: Keep track of what cleaning steps you performed
- **Preserve original data**: Always work on copies, never modify raw data
- **Understand your domain**: Missing value treatment should make sense in context
- **Validate results**: Check that cleaning didn't introduce new problems
- **Consider multiple strategies**: Compare different approaches (drop vs. fill)`
    },
    {
      title: "Descriptive Statistics and Data Summarization",
      content: `Descriptive statistics help you understand the central tendencies, variability, and distributions of your data.

\`\`\`python interactive
# Comprehensive statistical analysis
print("Descriptive Statistics Analysis:")
print("="*40)

# 1. Central tendency measures
print("1. Central Tendency Measures:")
numerical_cols = ['age', 'study_hours_per_week', 'previous_math_score', 
                 'attendance_rate', 'final_exam_score']

for col in numerical_cols:
    mean_val = df_clean[col].mean()
    median_val = df_clean[col].median()
    mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else "No mode"
    
    print(f"{col}:")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  Mode: {mode_val}")
    print()

# 2. Variability measures
print("2. Variability Measures:")
for col in numerical_cols:
    std_val = df_clean[col].std()
    var_val = df_clean[col].var()
    range_val = df_clean[col].max() - df_clean[col].min()
    iqr_val = df_clean[col].quantile(0.75) - df_clean[col].quantile(0.25)
    
    print(f"{col}:")
    print(f"  Standard Deviation: {std_val:.2f}")
    print(f"  Variance: {var_val:.2f}")
    print(f"  Range: {range_val:.2f}")
    print(f"  Interquartile Range: {iqr_val:.2f}")
    print()

# 3. Distribution shape
print("3. Distribution Shape:")
from scipy import stats

for col in numerical_cols:
    skewness = stats.skew(df_clean[col])
    kurtosis = stats.kurtosis(df_clean[col])
    
    print(f"{col}:")
    print(f"  Skewness: {skewness:.3f}", end="")
    if skewness > 0.5:
        print(" (right-skewed)")
    elif skewness < -0.5:
        print(" (left-skewed)")
    else:
        print(" (approximately symmetric)")
    
    print(f"  Kurtosis: {kurtosis:.3f}", end="")
    if kurtosis > 0:
        print(" (heavy-tailed)")
    else:
        print(" (light-tailed)")
    print()

# 4. Comprehensive summary statistics
print("4. Complete Statistical Summary:")
summary_stats = df_clean[numerical_cols].describe()
print(summary_stats)

# 5. Categorical variable analysis
print("\\n5. Categorical Variable Analysis:")
categorical_cols = ['course_grade', 'performance_category', 'study_intensity']

for col in categorical_cols:
    print(f"\\n{col} distribution:")
    value_counts = df_clean[col].value_counts()
    percentages = df_clean[col].value_counts(normalize=True) * 100
    
    for category in value_counts.index:
        count = value_counts[category]
        pct = percentages[category]
        print(f"  {category}: {count} ({pct:.1f}%)")

# 6. Cross-tabulation analysis
print("\\n6. Cross-tabulation Analysis:")
crosstab = pd.crosstab(df_clean['performance_category'], df_clean['study_intensity'])
print("Performance vs Study Intensity:")
print(crosstab)

# Calculate percentages
crosstab_pct = pd.crosstab(df_clean['performance_category'], df_clean['study_intensity'], normalize='index') * 100
print("\\nPercentages by performance category:")
print(crosstab_pct.round(1))
\`\`\`

**Key Statistical Concepts:**
- **Mean vs Median**: Mean affected by outliers, median more robust
- **Standard Deviation**: Measures spread in original units
- **Skewness**: Measures asymmetry of distribution
- **Kurtosis**: Measures tail heaviness
- **Quartiles**: Divide data into four equal parts (25%, 50%, 75%)`
    },
    {
      title: "Data Visualization and Pattern Discovery",
      content: `Effective visualization reveals patterns, trends, and relationships that might not be obvious from raw numbers alone.

\`\`\`python interactive
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plotting environment
plt.style.use('default')
sns.set_palette("husl")

print("Comprehensive Data Visualization:")
print("="*40)

# 1. Distribution Analysis
print("1. Creating Distribution Plots:")

# Create subplots for multiple visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Histogram of final exam scores
axes[0, 0].hist(df_clean['final_exam_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_xlabel('Final Exam Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Final Exam Scores')
axes[0, 0].grid(True, alpha=0.3)

# Box plot of study hours by performance category
performance_categories = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
study_hours_by_performance = [df_clean[df_clean['performance_category'] == cat]['study_hours_per_week'] 
                             for cat in performance_categories if cat in df_clean['performance_category'].values]

axes[0, 1].boxplot(study_hours_by_performance, labels=[cat for cat in performance_categories 
                                                      if cat in df_clean['performance_category'].values])
axes[0, 1].set_xlabel('Performance Category')
axes[0, 1].set_ylabel('Study Hours per Week')
axes[0, 1].set_title('Study Hours by Performance Category')
axes[0, 1].tick_params(axis='x', rotation=45)

# Scatter plot: study hours vs final exam score
axes[1, 0].scatter(df_clean['study_hours_per_week'], df_clean['final_exam_score'], 
                  alpha=0.6, color='green')
axes[1, 0].set_xlabel('Study Hours per Week')
axes[1, 0].set_ylabel('Final Exam Score')
axes[1, 0].set_title('Study Hours vs Final Exam Performance')
axes[1, 0].grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df_clean['study_hours_per_week'], df_clean['final_exam_score'], 1)
p = np.poly1d(z)
axes[1, 0].plot(df_clean['study_hours_per_week'], p(df_clean['study_hours_per_week']), 
               "r--", alpha=0.8, label=f'Trend line')
axes[1, 0].legend()

# Bar plot of course grade distribution
grade_counts = df_clean['course_grade'].value_counts()
axes[1, 1].bar(grade_counts.index, grade_counts.values, alpha=0.7, color='orange')
axes[1, 1].set_xlabel('Course Grade')
axes[1, 1].set_ylabel('Number of Students')
axes[1, 1].set_title('Distribution of Course Grades')

# Add value labels on bars
for i, v in enumerate(grade_counts.values):
    axes[1, 1].text(i, v + 1, str(v), ha='center')

plt.tight_layout()
plt.show()

# 2. Advanced visualizations using Seaborn
print("\\n2. Advanced Statistical Visualizations:")

# Create correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Correlation matrix
numeric_columns = ['age', 'study_hours_per_week', 'previous_math_score', 
                  'attendance_rate', 'final_exam_score']
correlation_matrix = df_clean[numeric_columns].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
           square=True, fmt='.2f', cbar_kws={"shrink": 0.8}, ax=axes[0])
axes[0].set_title('Correlation Matrix of Numeric Variables')

# Pair plot for key relationships
# Note: Using a subset for clarity
subset_data = df_clean[['study_hours_per_week', 'attendance_rate', 'final_exam_score']].sample(100)
scatter_matrix = pd.plotting.scatter_matrix(subset_data, alpha=0.6, figsize=(8, 6), 
                                          diagonal='hist', ax=axes[1])
axes[1].set_title('Pairwise Relationships (Sample of 100)')

plt.tight_layout()
plt.show()

# 3. Categorical analysis visualization
print("\\n3. Categorical Data Analysis:")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Stacked bar chart: performance by study intensity
crosstab = pd.crosstab(df_clean['study_intensity'], df_clean['performance_category'])
crosstab.plot(kind='bar', stacked=True, ax=axes[0], alpha=0.8)
axes[0].set_xlabel('Study Intensity')
axes[0].set_ylabel('Number of Students')
axes[0].set_title('Performance Distribution by Study Intensity')
axes[0].legend(title='Performance Category', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0].tick_params(axis='x', rotation=45)

# Violin plot: exam scores by course grade
grade_order = ['F', 'D', 'C', 'B', 'A']
grade_order = [g for g in grade_order if g in df_clean['course_grade'].unique()]

violin_data = [df_clean[df_clean['course_grade'] == grade]['final_exam_score'] 
              for grade in grade_order]
parts = axes[1].violinplot(violin_data, positions=range(len(grade_order)), showmeans=True)
axes[1].set_xticks(range(len(grade_order)))
axes[1].set_xticklabels(grade_order)
axes[1].set_xlabel('Course Grade')
axes[1].set_ylabel('Final Exam Score')
axes[1].set_title('Exam Score Distribution by Course Grade')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4. Statistical insights from visualizations
print("\\n4. Key Insights from Visualizations:")

# Calculate correlation between study hours and final exam score
correlation_study_exam = df_clean['study_hours_per_week'].corr(df_clean['final_exam_score'])
print(f"Correlation between study hours and exam score: {correlation_study_exam:.3f}")

if correlation_study_exam > 0.3:
    print("→ Strong positive relationship: more study time typically leads to higher scores")
elif correlation_study_exam > 0.1:
    print("→ Moderate positive relationship between study time and performance")
else:
    print("→ Weak relationship between study time and performance")

# Attendance analysis
high_attendance = df_clean[df_clean['attendance_rate'] > 0.9]['final_exam_score'].mean()
low_attendance = df_clean[df_clean['attendance_rate'] < 0.7]['final_exam_score'].mean()
print(f"\\nAverage exam score with high attendance (>90%): {high_attendance:.1f}")
print(f"Average exam score with low attendance (<70%): {low_attendance:.1f}")
print(f"Attendance impact: {high_attendance - low_attendance:.1f} point difference")

# Grade distribution insights
print(f"\\nGrade Distribution Analysis:")
total_students = len(df_clean)
for grade in ['A', 'B', 'C', 'D', 'F']:
    if grade in df_clean['course_grade'].values:
        count = (df_clean['course_grade'] == grade).sum()
        percentage = (count / total_students) * 100
        print(f"Grade {grade}: {count} students ({percentage:.1f}%)")
\`\`\`

**Visualization Best Practices:**
- **Choose appropriate chart types**: Histograms for distributions, scatter plots for relationships
- **Use color strategically**: Help distinguish categories and highlight patterns
- **Include clear labels**: Titles, axis labels, and legends are essential
- **Show uncertainty**: Error bars, confidence intervals when appropriate
- **Avoid chart junk**: Remove unnecessary elements that don't add information`
    },
    {
      title: "Introduction to Machine Learning with Scikit-learn",
      content: `Machine learning enables computers to automatically learn patterns from data and make predictions or decisions without being explicitly programmed for each specific task.

\`\`\`python interactive
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("Machine Learning with Scikit-learn:")
print("="*40)

# 1. Prepare data for machine learning
print("1. Data Preparation for ML:")

# Select features and target variable
# Features: factors that might influence final exam score
feature_columns = ['age', 'study_hours_per_week', 'previous_math_score', 'attendance_rate']
X = df_clean[feature_columns]
y = df_clean['final_exam_score']

print(f"Features selected: {feature_columns}")
print(f"Target variable: final_exam_score")
print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 2. Linear Regression Model
print("\\n2. Linear Regression Model:")

# Create and train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(lr_mse)
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

print(f"Linear Regression Performance:")
print(f"  Root Mean Squared Error: {lr_rmse:.2f}")
print(f"  Mean Absolute Error: {lr_mae:.2f}")
print(f"  R-squared Score: {lr_r2:.3f}")

# Interpret the model coefficients
print(f"\\nModel Interpretation:")
for feature, coef in zip(feature_columns, lr_model.coef_):
    print(f"  {feature}: {coef:.3f}")
    if 'hours' in feature and coef > 0:
        print(f"    → Each additional study hour increases exam score by {coef:.2f} points")
    elif 'attendance' in feature and coef > 0:
        print(f"    → Each 0.1 increase in attendance rate increases exam score by {coef*0.1:.2f} points")

print(f"  Intercept: {lr_model.intercept_:.2f}")

# 3. Random Forest Model (more advanced)
print("\\n3. Random Forest Model:")

# Create and train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

print(f"Random Forest Performance:")
print(f"  Root Mean Squared Error: {rf_rmse:.2f}")
print(f"  Mean Absolute Error: {rf_mae:.2f}")
print(f"  R-squared Score: {rf_r2:.3f}")

# Feature importance
print(f"\\nFeature Importance (Random Forest):")
feature_importance = rf_model.feature_importances_
for feature, importance in zip(feature_columns, feature_importance):
    print(f"  {feature}: {importance:.3f}")

# 4. Model Comparison
print("\\n4. Model Comparison:")
print(f"Linear Regression R²: {lr_r2:.3f}")
print(f"Random Forest R²: {rf_r2:.3f}")

if rf_r2 > lr_r2:
    print("→ Random Forest performs better (captures non-linear relationships)")
else:
    print("→ Linear Regression performs better (simpler, more interpretable)")

# 5. Making predictions for new students
print("\\n5. Predictions for New Students:")

# Create profiles for hypothetical students
new_students = pd.DataFrame({
    'age': [19, 21, 20],
    'study_hours_per_week': [10, 25, 15],
    'previous_math_score': [70, 85, 80],
    'attendance_rate': [0.8, 0.95, 0.9]
})

# Make predictions with both models
lr_predictions = lr_model.predict(new_students)
rf_predictions = rf_model.predict(new_students)

print("Student profiles and predictions:")
for i in range(len(new_students)):
    print(f"\\nStudent {i+1}:")
    print(f"  Age: {new_students.iloc[i]['age']}")
    print(f"  Study hours/week: {new_students.iloc[i]['study_hours_per_week']}")
    print(f"  Previous math score: {new_students.iloc[i]['previous_math_score']}")
    print(f"  Attendance rate: {new_students.iloc[i]['attendance_rate']}")
    print(f"  Linear Regression prediction: {lr_predictions[i]:.1f}")
    print(f"  Random Forest prediction: {rf_predictions[i]:.1f}")

# 6. Classification Example
print("\\n6. Classification Example - Predicting Course Grades:")

# Convert course grades to binary classification (Pass/Fail)
df_clean['pass_fail'] = df_clean['course_grade'].apply(lambda x: 1 if x in ['A', 'B', 'C'] else 0)

# Prepare data for classification
X_class = df_clean[feature_columns]
y_class = df_clean['pass_fail']

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

# Train a classification model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_class, y_train_class)

# Make predictions and evaluate
y_pred_class = clf_model.predict(X_test_class)
accuracy = accuracy_score(y_test_class, y_pred_class)

print(f"Classification Accuracy: {accuracy:.3f}")
print(f"This means the model correctly predicts pass/fail {accuracy*100:.1f}% of the time")

# Predict pass/fail for our new students
pass_fail_predictions = clf_model.predict(new_students)
pass_probabilities = clf_model.predict_proba(new_students)

print("\\nPass/Fail predictions for new students:")
for i in range(len(new_students)):
    prediction = "Pass" if pass_fail_predictions[i] == 1 else "Fail"
    confidence = max(pass_probabilities[i]) * 100
    print(f"Student {i+1}: {prediction} (confidence: {confidence:.1f}%)")
\`\`\`

**Key Machine Learning Concepts:**
- **Supervised Learning**: Learning from labeled examples (input-output pairs)
- **Training vs Testing**: Use separate data to evaluate model performance
- **Feature Selection**: Choosing relevant input variables
- **Model Evaluation**: Metrics like R², RMSE, accuracy to assess performance
- **Overfitting**: When model performs well on training data but poorly on new data`
    },
    {
      title: "Correlation Analysis and Statistical Relationships",
      content: `Understanding relationships between variables is crucial for data science. Correlation analysis helps identify which factors are related and how strongly.

\`\`\`python interactive
from scipy.stats import pearsonr, spearmanr
import numpy as np

print("Correlation Analysis and Statistical Relationships:")
print("="*50)

# 1. Correlation Matrix Analysis
print("1. Comprehensive Correlation Analysis:")

# Calculate correlation matrix
numeric_columns = ['age', 'study_hours_per_week', 'previous_math_score', 
                  'attendance_rate', 'final_exam_score']
correlation_matrix = df_clean[numeric_columns].corr()

print("Correlation Matrix:")
print(correlation_matrix.round(3))

# 2. Detailed correlation analysis with significance testing
print("\\n2. Statistical Significance of Correlations:")

for i, var1 in enumerate(numeric_columns):
    for j, var2 in enumerate(numeric_columns):
        if i < j:  # Only upper triangle to avoid duplicates
            corr_coef, p_value = pearsonr(df_clean[var1], df_clean[var2])
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            print(f"{var1} vs {var2}:")
            print(f"  Correlation: {corr_coef:.3f} {significance}")
            print(f"  P-value: {p_value:.4f}")
            
            # Interpret correlation strength
            if abs(corr_coef) >= 0.7:
                strength = "very strong"
            elif abs(corr_coef) >= 0.5:
                strength = "strong"
            elif abs(corr_coef) >= 0.3:
                strength = "moderate"
            elif abs(corr_coef) >= 0.1:
                strength = "weak"
            else:
                strength = "very weak"
            
            direction = "positive" if corr_coef > 0 else "negative"
            print(f"  Interpretation: {strength} {direction} relationship")
            print()

# 3. Advanced relationship analysis
print("3. Advanced Relationship Analysis:")

# Spearman correlation (non-parametric, captures monotonic relationships)
spearman_corr = df_clean[numeric_columns].corr(method='spearman')
print("Spearman Rank Correlation (captures non-linear monotonic relationships):")
print(spearman_corr.round(3))

# 4. Conditional analysis
print("\\n4. Conditional Relationship Analysis:")

# Analyze correlation within subgroups
high_performers = df_clean[df_clean['final_exam_score'] >= 80]
low_performers = df_clean[df_clean['final_exam_score'] < 70]

print("Correlation: Study Hours vs Attendance Rate")
print(f"High performers (score ≥80): {high_performers['study_hours_per_week'].corr(high_performers['attendance_rate']):.3f}")
print(f"Low performers (score <70): {low_performers['study_hours_per_week'].corr(low_performers['attendance_rate']):.3f}")

# 5. Regression analysis for understanding relationships
print("\\n5. Regression Analysis for Causal Understanding:")

# Multiple regression to understand relative importance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Standardize features for fair comparison of coefficients
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[['study_hours_per_week', 'previous_math_score', 'attendance_rate']])
y = df_clean['final_exam_score']

# Fit multiple regression
mr_model = LinearRegression()
mr_model.fit(X_scaled, y)

print("Multiple Regression Results (standardized coefficients):")
feature_names = ['study_hours_per_week', 'previous_math_score', 'attendance_rate']
for feature, coef in zip(feature_names, mr_model.coef_):
    print(f"  {feature}: {coef:.3f}")

print(f"\\nR-squared: {mr_model.score(X_scaled, y):.3f}")
print("Note: Larger absolute coefficient = stronger influence on final exam score")

# 6. Practical implications
print("\\n6. Practical Implications for Students:")

# Find the most important factors
abs_coefficients = np.abs(mr_model.coef_)
most_important_idx = np.argmax(abs_coefficients)
most_important_factor = feature_names[most_important_idx]

print(f"Most important factor for exam performance: {most_important_factor}")

# Calculate effect sizes
study_effect = mr_model.coef_[0] * np.std(df_clean['study_hours_per_week'])
attendance_effect = mr_model.coef_[2] * np.std(df_clean['attendance_rate'])

print(f"\\nPractical effect sizes:")
print(f"Increasing study hours by 1 std dev (≈{np.std(df_clean['study_hours_per_week']):.1f} hours): {study_effect:.1f} point increase")
print(f"Increasing attendance by 1 std dev (≈{np.std(df_clean['attendance_rate']):.2f}): {attendance_effect:.1f} point increase")

# Recommendations based on analysis
print(f"\\n7. Data-Driven Recommendations:")
print("Based on our correlation and regression analysis:")

if mr_model.coef_[0] > 0:
    print("• Increasing study time has a positive impact on exam performance")
if mr_model.coef_[2] > 0:
    print("• Regular class attendance significantly improves outcomes")
if mr_model.coef_[1] > 0:
    print("• Strong foundation (previous math score) predicts future success")

print("• These relationships suggest that student effort and engagement are key factors")
print("• Academic success appears to be influenced by multiple controllable factors")
\`\`\`

**Important Statistical Concepts:**
- **Correlation ≠ Causation**: Strong correlation doesn't prove one causes the other
- **P-values**: Probability that observed correlation occurred by chance
- **Effect Size**: Magnitude of relationship (beyond just statistical significance)
- **Multiple Regression**: Controls for other variables to isolate individual effects
- **Standardized Coefficients**: Allow comparison of variables with different units`
    },
    {
      title: "Real-World Applications and Career Preparation",
      content: `Data science skills are increasingly valuable across industries. Understanding how to apply these techniques to real-world problems opens many career opportunities.

**Industry Applications:**

**🏥 Healthcare and Medicine**
- **Patient Outcome Prediction**: Using patient data to predict treatment success rates
- **Drug Discovery**: Analyzing molecular data to identify potential medications
- **Epidemic Modeling**: Tracking disease spread patterns using public health data
- **Medical Imaging**: Computer vision for X-ray and MRI analysis
- **Personalized Medicine**: Tailoring treatments based on genetic and lifestyle data

**💰 Finance and Banking**
- **Credit Scoring**: Predicting loan default risk using customer financial history
- **Fraud Detection**: Identifying suspicious transactions in real-time
- **Algorithmic Trading**: Using market data to make automated investment decisions
- **Risk Management**: Modeling market volatility and portfolio risk
- **Customer Segmentation**: Targeted marketing for financial products

**🛒 E-commerce and Retail**
- **Recommendation Systems**: "Customers who bought this also bought..."
- **Price Optimization**: Dynamic pricing based on demand and competition
- **Inventory Management**: Predicting demand to optimize stock levels
- **Customer Lifetime Value**: Estimating long-term customer profitability
- **A/B Testing**: Comparing different website designs or marketing strategies

**🚗 Transportation and Logistics**
- **Route Optimization**: Finding fastest delivery paths using traffic data
- **Autonomous Vehicles**: Machine learning for self-driving car navigation
- **Predictive Maintenance**: Preventing vehicle breakdowns before they occur
- **Demand Forecasting**: Predicting ride-sharing or public transport needs
- **Supply Chain Analytics**: Optimizing global shipping and distribution

**Skills Development Roadmap:**

**📚 Undergraduate Focus Areas:**
1. **Programming Proficiency**
   - Master Python for data science (pandas, NumPy, scikit-learn, matplotlib)
   - Learn SQL for database queries and data extraction
   - Version control with Git and collaborative development

2. **Statistical Foundation**
   - Descriptive and inferential statistics
   - Hypothesis testing and experimental design
   - Regression analysis and correlation interpretation
   - Understanding of probability distributions

3. **Machine Learning Basics**
   - Supervised learning (classification and regression)
   - Unsupervised learning (clustering and dimensionality reduction)
   - Model evaluation and validation techniques
   - Understanding bias-variance tradeoff

4. **Data Visualization**
   - Creating effective charts and graphs
   - Dashboard design principles
   - Storytelling with data
   - Interactive visualization tools

**🎯 Practical Experience:**

**Personal Projects**
- Analyze publicly available datasets (sports, economics, social media)
- Participate in Kaggle competitions to practice real-world problems
- Create a portfolio showcasing different types of analysis
- Write blog posts explaining your findings and methodology

**Academic Opportunities**
- Research projects with faculty members
- Data analysis for student organizations or local businesses
- Internships with companies that use data science
- Independent study courses focused on specific applications

**Professional Development**
- Join data science clubs and attend meetups
- Contribute to open-source data science projects
- Attend conferences and workshops (virtual or in-person)
- Network with professionals in the field

**🚀 Career Paths:**

**Data Analyst**
- Entry-level position focusing on descriptive analytics
- Create reports and dashboards for business stakeholders
- Required skills: SQL, Excel, basic programming, visualization tools
- Typical salary: $45,000-$65,000 (entry level)

**Data Scientist**
- Build predictive models and conduct advanced analytics
- Requires strong programming and machine learning skills
- Work on complex business problems requiring statistical inference
- Typical salary: $70,000-$120,000 (mid-level)

**Machine Learning Engineer**
- Deploy models into production systems
- Focus on scalability, performance, and software engineering
- Bridge between data science and software development
- Typical salary: $90,000-$150,000

**Business Intelligence Analyst**
- Focus on business metrics and performance measurement
- Create executive dashboards and strategic reports
- Strong business acumen combined with technical skills
- Typical salary: $55,000-$85,000

**Research Scientist**
- Academic or industry research positions
- Develop new methodologies and algorithms
- Publish research papers and advance the field
- Requires advanced degree (Master's/PhD)

**Next Steps Action Plan:**

**Immediate (Next Semester):**
- [ ] Complete additional statistics or computer science courses
- [ ] Start a personal data science project using real data
- [ ] Learn advanced pandas and data manipulation techniques
- [ ] Practice with different types of datasets and problems

**Short-term (Next Year):**
- [ ] Apply for data science internships
- [ ] Build a portfolio with 3-5 complete projects
- [ ] Learn additional tools (R, Tableau, SQL databases)
- [ ] Participate in at least one Kaggle competition

**Medium-term (2-3 Years):**
- [ ] Complete specialized courses in machine learning or statistics
- [ ] Gain practical experience through internships or part-time work
- [ ] Develop expertise in a specific domain (healthcare, finance, etc.)
- [ ] Consider pursuing a graduate degree for advanced positions

 The field of data science continues to grow rapidly, offering excellent career prospects for those who develop strong analytical and technical skills. The foundation you've built in this lesson provides the groundwork for a successful career in this exciting and impactful field.`
    }
  ]
};
