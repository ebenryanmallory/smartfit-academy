// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.

import type { LessonData } from "../types";

export const lesson: LessonData = {
  id: "c-data-science-fundamentals",
  title: "Data Science Fundamentals",
  description: "Learn data science concepts, statistical analysis, and data visualization with hands-on Python programming for real-world applications.",
  sections: [
    {
      title: "Introduction to Data Science",
      content: `Data Science is an interdisciplinary field that combines programming, statistics, and domain expertise to extract meaningful insights from data.

**Core Components of Data Science:**
- **Data Collection**: Gathering data from various sources (databases, APIs, surveys, sensors)
- **Data Cleaning**: Preprocessing raw data to handle missing values, errors, and inconsistencies
- **Exploratory Data Analysis (EDA)**: Understanding data patterns through visualization and statistics
- **Statistical Modeling**: Building mathematical models to understand relationships and make predictions
- **Communication**: Presenting findings through visualizations, reports, and dashboards

**Why Data Science Matters:**
- **Evidence-Based Decisions**: Organizations use data to make informed choices rather than guessing
- **Personalization**: Companies create customized experiences (Netflix recommendations, targeted ads)
- **Scientific Discovery**: Researchers use data science to understand complex phenomena
- **Social Impact**: Data helps solve problems in healthcare, education, climate change, and social justice

**Data Science vs Related Fields:**
- **Statistics**: Focus on mathematical theory and inference
- **Computer Science**: Emphasis on algorithms and computational efficiency
- **Machine Learning**: Subset focused on predictive modeling and pattern recognition
- **Business Analytics**: Application of data science to business problems`
    },
    {
      title: "Working with Data: Pandas Fundamentals",
      content: `Pandas is the most important Python library for data manipulation and analysis. It provides powerful data structures for handling structured data.

\`\`\`python interactive
import pandas as pd
import numpy as np

# Create a sample dataset of student performance
data = {
    'student_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eva', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
    'math_score': [85, 92, 78, 96, 89, 73, 91, 87, 94, 82],
    'science_score': [88, 87, 82, 93, 91, 76, 89, 85, 97, 79],
    'english_score': [92, 85, 89, 88, 93, 81, 94, 90, 89, 86],
    'hours_studied': [6, 8, 5, 9, 7, 4, 8, 6, 9, 5],
    'grade_level': [10, 11, 9, 12, 11, 9, 12, 10, 11, 10]
}

# Create DataFrame
df = pd.DataFrame(data)
print("Student Performance Dataset:")
print(df)

# Basic DataFrame operations
print(f"\\nDataset shape: {df.shape}")
print(f"Column names: {list(df.columns)}")
print(f"Data types:\\n{df.dtypes}")

# Statistical summary
print("\\nStatistical Summary:")
print(df.describe())

# Calculate total score and GPA
df['total_score'] = df['math_score'] + df['science_score'] + df['english_score']
df['gpa'] = df['total_score'] / 30  # Assuming 100 point scale converts to 10 point GPA

print("\\nDataset with calculated fields:")
print(df[['name', 'total_score', 'gpa', 'hours_studied']].head())
\`\`\``
    },
    {
      title: "Data Cleaning and Preprocessing",
      content: `Real-world data is messy. Data cleaning is often 80% of a data scientist's work.

\`\`\`python interactive
# Create messy dataset with common problems
messy_data = {
    'student_name': ['Alice Smith', 'bob jones', 'CHARLIE BROWN', None, 'Eva Davis', 'Frank Miller'],
    'test_score': [85, 92, '78', 96, None, 'invalid'],
    'attendance': [0.95, 1.2, 0.87, 0.92, 0.89, 0.76],  # 1.2 is impossible (>100%)
    'grade': ['A', 'A-', 'B+', 'A', 'B', 'C+'],
    'graduation_year': [2024, 2024, 2025, 2024, 2023, 2024]
}

messy_df = pd.DataFrame(messy_data)
print("Original messy data:")
print(messy_df)
print(f"\\nData types:\\n{messy_df.dtypes}")

# Data cleaning steps
print("\\n=== DATA CLEANING PROCESS ===")

# 1. Handle missing values
print("\\n1. Missing values before cleaning:")
print(messy_df.isnull().sum())

# Fill missing name with 'Unknown'
messy_df['student_name'] = messy_df['student_name'].fillna('Unknown Student')

# 2. Clean and convert data types
# Convert test_score to numeric, invalid values become NaN
messy_df['test_score'] = pd.to_numeric(messy_df['test_score'], errors='coerce')

# Fill missing test scores with median
median_score = messy_df['test_score'].median()
messy_df['test_score'] = messy_df['test_score'].fillna(median_score)

print(f"\\n2. Filled missing test score with median: {median_score}")

# 3. Fix data inconsistencies
# Standardize name formatting
messy_df['student_name'] = messy_df['student_name'].str.title()

# Fix impossible attendance values (cap at 1.0)
messy_df['attendance'] = messy_df['attendance'].clip(upper=1.0)

# 4. Create categorical variables
grade_mapping = {'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7, 'C+': 2.3, 'C': 2.0}
messy_df['grade_points'] = messy_df['grade'].map(grade_mapping)

print("\\nCleaned dataset:")
print(messy_df)

# 5. Data validation
print("\\n=== DATA VALIDATION ===")
print(f"Test scores range: {messy_df['test_score'].min()} - {messy_df['test_score'].max()}")
print(f"Attendance range: {messy_df['attendance'].min()} - {messy_df['attendance'].max()}")
print(f"Missing values after cleaning:\\n{messy_df.isnull().sum()}")
\`\`\``
    },
    {
      title: "Exploratory Data Analysis (EDA)",
      content: `EDA helps you understand your data before building models. It involves calculating statistics and creating visualizations.

\`\`\`python interactive
import matplotlib.pyplot as plt

# Extend our student dataset for analysis
np.random.seed(42)
n_students = 50

# Generate synthetic student data
student_data = pd.DataFrame({
    'math_score': np.random.normal(82, 12, n_students).round().astype(int),
    'science_score': np.random.normal(80, 15, n_students).round().astype(int),
    'english_score': np.random.normal(85, 10, n_students).round().astype(int),
    'hours_studied': np.random.normal(6, 2, n_students).round(1),
    'extracurriculars': np.random.choice(['Sports', 'Music', 'Drama', 'None'], n_students),
    'grade_level': np.random.choice([9, 10, 11, 12], n_students)
})

# Ensure realistic score ranges
student_data['math_score'] = student_data['math_score'].clip(0, 100)
student_data['science_score'] = student_data['science_score'].clip(0, 100)
student_data['english_score'] = student_data['english_score'].clip(0, 100)
student_data['hours_studied'] = student_data['hours_studied'].clip(0, 12)

print("=== EXPLORATORY DATA ANALYSIS ===")

# 1. Descriptive Statistics
print("\\n1. DESCRIPTIVE STATISTICS")
print(student_data.describe())

# 2. Correlation Analysis
print("\\n2. CORRELATION ANALYSIS")
correlation_matrix = student_data[['math_score', 'science_score', 'english_score', 'hours_studied']].corr()
print(correlation_matrix.round(3))

# 3. Group Analysis
print("\\n3. GROUP ANALYSIS")
print("Average scores by extracurricular activity:")
extracurricular_stats = student_data.groupby('extracurriculars').agg({
    'math_score': 'mean',
    'science_score': 'mean', 
    'english_score': 'mean',
    'hours_studied': 'mean'
}).round(1)
print(extracurricular_stats)

# 4. Grade Level Analysis
print("\\nAverage scores by grade level:")
grade_stats = student_data.groupby('grade_level').agg({
    'math_score': ['mean', 'std'],
    'hours_studied': 'mean'
}).round(2)
print(grade_stats)

# 5. Find interesting patterns
print("\\n4. INTERESTING PATTERNS")

# Students who study a lot vs. little
high_study = student_data[student_data['hours_studied'] > 7]
low_study = student_data[student_data['hours_studied'] < 4]

print(f"High study group (>7 hrs): {len(high_study)} students")
print(f"  Average math score: {high_study['math_score'].mean():.1f}")
print(f"Low study group (<4 hrs): {len(low_study)} students") 
print(f"  Average math score: {low_study['math_score'].mean():.1f}")

# Find top performers
student_data['total_score'] = student_data['math_score'] + student_data['science_score'] + student_data['english_score']
top_10_percent = student_data.nlargest(5, 'total_score')

print(f"\\nTop 5 students by total score:")
print(top_10_percent[['math_score', 'science_score', 'english_score', 'total_score', 'hours_studied']])
\`\`\``
    },
    {
      title: "Data Visualization with Matplotlib",
      content: `Visualizations help communicate insights and identify patterns that aren't obvious in raw numbers.

\`\`\`python interactive
import matplotlib.pyplot as plt

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Student Performance Analysis', fontsize=16)

# 1. Histogram of math scores
axes[0, 0].hist(student_data['math_score'], bins=15, color='skyblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Distribution of Math Scores')
axes[0, 0].set_xlabel('Math Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(student_data['math_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {student_data["math_score"].mean():.1f}')
axes[0, 0].legend()

# 2. Scatter plot: Study hours vs Math score
axes[0, 1].scatter(student_data['hours_studied'], student_data['math_score'], 
                   alpha=0.6, color='green')
axes[0, 1].set_title('Study Hours vs Math Score')
axes[0, 1].set_xlabel('Hours Studied')
axes[0, 1].set_ylabel('Math Score')

# Add trend line
z = np.polyfit(student_data['hours_studied'], student_data['math_score'], 1)
p = np.poly1d(z)
axes[0, 1].plot(student_data['hours_studied'], p(student_data['hours_studied']), 
                "r--", alpha=0.8, label=f'Trend line')
axes[0, 1].legend()

# 3. Box plot by extracurricular activity
extracurricular_groups = [student_data[student_data['extracurriculars'] == activity]['math_score'].values 
                         for activity in student_data['extracurriculars'].unique()]
axes[1, 0].boxplot(extracurricular_groups, labels=student_data['extracurriculars'].unique())
axes[1, 0].set_title('Math Scores by Extracurricular Activity')
axes[1, 0].set_xlabel('Activity')
axes[1, 0].set_ylabel('Math Score')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Grade level performance
grade_means = student_data.groupby('grade_level')['math_score'].mean()
axes[1, 1].bar(grade_means.index, grade_means.values, color='orange', alpha=0.7)
axes[1, 1].set_title('Average Math Score by Grade Level')
axes[1, 1].set_xlabel('Grade Level')
axes[1, 1].set_ylabel('Average Math Score')

# Add value labels on bars
for i, v in enumerate(grade_means.values):
    axes[1, 1].text(grade_means.index[i], v + 1, f'{v:.1f}', ha='center')

plt.tight_layout()
plt.show()

# Statistical insights from visualizations
print("\\n=== INSIGHTS FROM VISUALIZATIONS ===")
print(f"1. Math score distribution appears roughly normal with mean {student_data['math_score'].mean():.1f}")

# Calculate correlation between study hours and math score
correlation = student_data['hours_studied'].corr(student_data['math_score'])
print(f"2. Correlation between study hours and math score: {correlation:.3f}")

if correlation > 0.3:
    print("   → Strong positive relationship: more study time = higher scores")
elif correlation > 0.1:
    print("   → Moderate positive relationship")
else:
    print("   → Weak relationship")

# Find best performing extracurricular group
best_activity = student_data.groupby('extracurriculars')['math_score'].mean().idxmax()
best_score = student_data.groupby('extracurriculars')['math_score'].mean().max()
print(f"3. Best performing extracurricular group: {best_activity} (avg: {best_score:.1f})")
\`\`\``
    },
    {
      title: "Statistical Analysis and Hypothesis Testing",
      content: `Statistical testing helps determine if observed differences are significant or just due to random chance.

\`\`\`python interactive
from scipy import stats

print("=== STATISTICAL HYPOTHESIS TESTING ===")

# Question 1: Do students who do sports score differently than those who don't?
sports_students = student_data[student_data['extracurriculars'] == 'Sports']['math_score']
non_sports_students = student_data[student_data['extracurriculars'] != 'Sports']['math_score']

print("\\n1. COMPARING SPORTS vs NON-SPORTS STUDENTS")
print(f"Sports students (n={len(sports_students)}): Mean = {sports_students.mean():.1f}, SD = {sports_students.std():.1f}")
print(f"Non-sports students (n={len(non_sports_students)}): Mean = {non_sports_students.mean():.1f}, SD = {non_sports_students.std():.1f}")

# Perform t-test
t_stat, p_value = stats.ttest_ind(sports_students, non_sports_students)
print(f"\\nT-test results:")
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.3f}")

alpha = 0.05
if p_value < alpha:
    print(f"Result: Significant difference (p < {alpha})")
else:
    print(f"Result: No significant difference (p >= {alpha})")

# Question 2: Is there a relationship between study hours and total score?
total_scores = student_data['math_score'] + student_data['science_score'] + student_data['english_score']
correlation_coeff, correlation_p = stats.pearsonr(student_data['hours_studied'], total_scores)

print("\\n2. CORRELATION: STUDY HOURS vs TOTAL SCORE")
print(f"Correlation coefficient: {correlation_coeff:.3f}")
print(f"P-value: {correlation_p:.3f}")

if correlation_p < 0.05:
    if correlation_coeff > 0:
        print("Result: Significant positive correlation - more study time relates to higher scores")
    else:
        print("Result: Significant negative correlation")
else:
    print("Result: No significant correlation")

# Question 3: ANOVA - Do all grade levels perform equally?
grade_9 = student_data[student_data['grade_level'] == 9]['math_score']
grade_10 = student_data[student_data['grade_level'] == 10]['math_score']
grade_11 = student_data[student_data['grade_level'] == 11]['math_score']
grade_12 = student_data[student_data['grade_level'] == 12]['math_score']

print("\\n3. ANOVA: COMPARING ALL GRADE LEVELS")
f_stat, anova_p = stats.f_oneway(grade_9, grade_10, grade_11, grade_12)
print(f"F-statistic: {f_stat:.3f}")
print(f"P-value: {anova_p:.3f}")

if anova_p < 0.05:
    print("Result: Significant differences between grade levels")
    
    # Which grades are different?
    grade_means = student_data.groupby('grade_level')['math_score'].mean()
    print("\\nGrade level means:")
    for grade, mean in grade_means.items():
        print(f"  Grade {grade}: {mean:.1f}")
else:
    print("Result: No significant differences between grade levels")

# Confidence Intervals
print("\\n4. CONFIDENCE INTERVALS")
mean_math = student_data['math_score'].mean()
sem_math = stats.sem(student_data['math_score'])  # Standard error of mean
ci_95 = stats.t.interval(0.95, len(student_data)-1, mean_math, sem_math)

print(f"Math score 95% confidence interval: {ci_95[0]:.1f} to {ci_95[1]:.1f}")
print(f"Interpretation: We're 95% confident the true population mean is between {ci_95[0]:.1f} and {ci_95[1]:.1f}")
\`\`\``
    },
    {
      title: "Building Predictive Models",
      content: `Use statistical models to predict outcomes based on input variables.

\`\`\`python interactive
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=== PREDICTIVE MODELING ===")

# Prepare data for modeling
# Predict total score based on study hours, grade level, and extracurricular activity
student_data['total_score'] = student_data['math_score'] + student_data['science_score'] + student_data['english_score']

# Create dummy variables for categorical data
student_modeling = pd.get_dummies(student_data, columns=['extracurriculars'], prefix='activity')

# Features for prediction
feature_columns = ['hours_studied', 'grade_level'] + [col for col in student_modeling.columns if col.startswith('activity_')]
X = student_modeling[feature_columns]
y = student_modeling['total_score']

print(f"Features used for prediction: {feature_columns}")
print(f"Dataset shape: {X.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set: {X_train.shape[0]} students")
print(f"Test set: {X_test.shape[0]} students")

# Model 1: Linear Regression
print("\\n1. LINEAR REGRESSION MODEL")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
lr_predictions = lr_model.predict(X_test)

# Evaluate model
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

print(f"Linear Regression Results:")
print(f"  Mean Squared Error: {lr_mse:.1f}")
print(f"  R-squared Score: {lr_r2:.3f}")
print(f"  Root Mean Squared Error: {np.sqrt(lr_mse):.1f} points")

# Feature importance (coefficients)
print(f"\\n  Feature Importance (coefficients):")
for feature, coef in zip(feature_columns, lr_model.coef_):
    print(f"    {feature}: {coef:.2f}")

# Model 2: Random Forest
print("\\n2. RANDOM FOREST MODEL")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print(f"Random Forest Results:")
print(f"  Mean Squared Error: {rf_mse:.1f}")
print(f"  R-squared Score: {rf_r2:.3f}")
print(f"  Root Mean Squared Error: {np.sqrt(rf_mse):.1f} points")

# Feature importance
print(f"\\n  Feature Importance:")
for feature, importance in zip(feature_columns, rf_model.feature_importances_):
    print(f"    {feature}: {importance:.3f}")

# Model comparison
print("\\n3. MODEL COMPARISON")
print(f"Linear Regression R²: {lr_r2:.3f}")
print(f"Random Forest R²: {rf_r2:.3f}")

if rf_r2 > lr_r2:
    print("→ Random Forest performs better")
    best_model = rf_model
    best_predictions = rf_predictions
else:
    print("→ Linear Regression performs better")
    best_model = lr_model
    best_predictions = lr_predictions

# Make predictions for new students
print("\\n4. PREDICTIONS FOR NEW STUDENTS")
new_students = pd.DataFrame({
    'hours_studied': [5, 8, 3],
    'grade_level': [10, 11, 9],
    'activity_Drama': [1, 0, 0],
    'activity_Music': [0, 1, 0], 
    'activity_None': [0, 0, 1],
    'activity_Sports': [0, 0, 0]
})

new_predictions = best_model.predict(new_students)

for i, pred in enumerate(new_predictions):
    hours = new_students.iloc[i]['hours_studied']
    grade = new_students.iloc[i]['grade_level']
    print(f"Student {i+1}: {hours} hours study, Grade {grade} → Predicted total: {pred:.0f}")
\`\`\``
    },
    {
      title: "Real-World Applications and Career Preparation",
      content: `Data science is transforming every industry and creating exciting career opportunities.

**Industry Applications:**

**🏥 Healthcare & Medicine**
- Analyzing medical records to improve treatment outcomes
- Drug discovery using molecular data analysis
- Predicting disease outbreaks using epidemiological data
- Personalized medicine based on genetic profiles

**📈 Business & Finance**
- Customer segmentation for targeted marketing
- Fraud detection in financial transactions
- Supply chain optimization and demand forecasting
- Risk assessment for loans and investments

**🌱 Environmental Science**
- Climate change modeling and prediction
- Tracking biodiversity and species conservation
- Optimizing renewable energy systems
- Monitoring air and water quality

**🎓 Education & Social Impact**
- Personalizing learning experiences for students
- Analyzing educational outcomes to improve teaching methods
- Identifying at-risk students for early intervention
- Optimizing school resource allocation

**🚗 Technology & Innovation**
- Recommendation systems (Netflix, Spotify, Amazon)
- Autonomous vehicle navigation and safety
- Natural language processing for chatbots
- Computer vision for medical imaging

**Career Preparation Path:**

**📚 Academic Foundation (High School):**
- **Mathematics**: Strong foundation in algebra, statistics, and basic calculus
- **Computer Science**: Programming skills (Python, R), database concepts
- **Science**: Understanding of scientific method and experimental design
- **Communication**: Writing and presentation skills for sharing insights

**💻 Technical Skills to Develop:**
- **Programming**: Python (pandas, scikit-learn, matplotlib), SQL for databases
- **Statistics**: Hypothesis testing, regression analysis, probability
- **Visualization**: Creating clear, compelling charts and dashboards
- **Machine Learning**: Understanding when and how to apply different algorithms

**🎯 Practical Experience:**
- **Personal Projects**: Analyze datasets that interest you (sports, music, social media)
- **Competitions**: Participate in Kaggle competitions or science fairs
- **Internships**: Seek summer opportunities with local businesses or research labs
- **Online Courses**: Coursera, edX, Khan Academy for structured learning

**🔮 Future Outlook:**
- Data Science is one of the fastest-growing career fields
- Median salary for data scientists is significantly above average
- High demand across all industries and sectors
- Excellent work-life balance and remote work opportunities
- Continuous learning keeps the work interesting and challenging

**Next Steps Action Plan:**
1. **Master the fundamentals**: Excel in math and computer science courses
2. **Build a portfolio**: Create 3-5 data science projects showcasing different skills
3. **Network**: Join local data science meetups or online communities
4. **Stay current**: Follow data science blogs, podcasts, and news
5. **Consider college majors**: Statistics, Computer Science, Mathematics, or specialized Data Science programs

The combination of technical skills, analytical thinking, and domain expertise makes data science both intellectually rewarding and financially attractive!`
    }
  ]
}; 