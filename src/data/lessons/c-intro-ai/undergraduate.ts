// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.

import type { LessonData } from "../types";

export const lesson: LessonData = {
  id: "c-intro-ai",
  title: "Introduction to Artificial Intelligence",
  description: "Comprehensive introduction to AI fundamentals, methodologies, and practical applications for undergraduate students.",
  sections: [
    {
      title: "Understanding Artificial Intelligence",
      content: `Artificial Intelligence (AI) represents one of the most significant technological developments of our time. At its core, AI refers to computer systems capable of performing tasks that typically require human intelligence, including learning, reasoning, problem-solving, perception, and language understanding.

**Defining Intelligence in Machines**
Intelligence in AI systems manifests through several key capabilities:
- **Perception**: Processing and interpreting sensory data (vision, speech, text)
- **Learning**: Acquiring knowledge and skills from experience and data
- **Reasoning**: Drawing logical conclusions and making inferences
- **Planning**: Developing strategies to achieve specific goals
- **Communication**: Understanding and generating natural language

**Historical Context and Evolution**
The field of AI emerged in the 1950s with pioneers like Alan Turing, who proposed the famous "Turing Test" as a benchmark for machine intelligence. The journey from rule-based expert systems to today's neural networks represents decades of research breakthroughs, computational advances, and algorithmic innovations.

**AI vs. Human Intelligence**
While AI excels in specific domains like pattern recognition and data analysis, human intelligence remains superior in areas requiring creativity, emotional understanding, and contextual reasoning. Modern AI systems are "narrow" in scope, designed for specific tasks rather than general-purpose reasoning.`
    },
    {
      title: "Core AI Methodologies and Approaches",
      content: `Understanding AI requires familiarity with its fundamental methodologies, each suited to different types of problems and data scenarios.

**Machine Learning Paradigms**

*Supervised Learning*
- Learns from labeled training data to make predictions on new inputs
- Applications: Email classification, medical diagnosis, stock price prediction
- Algorithms: Linear regression, decision trees, support vector machines
- Challenge: Requires large amounts of labeled data

*Unsupervised Learning*
- Discovers hidden patterns in data without explicit labels
- Applications: Customer segmentation, anomaly detection, data compression
- Algorithms: K-means clustering, principal component analysis, autoencoders
- Challenge: Difficult to evaluate results without ground truth

*Reinforcement Learning*
- Learns optimal actions through interaction with an environment
- Applications: Game playing, robotics, autonomous vehicles, recommendation systems
- Key concepts: Rewards, states, actions, policies
- Challenge: Balancing exploration vs. exploitation

**Deep Learning and Neural Networks**
Deep learning uses multi-layered neural networks to automatically learn hierarchical representations from data. Key architectures include:
- **Feedforward Networks**: Basic building blocks for pattern recognition
- **Convolutional Neural Networks (CNNs)**: Specialized for image processing
- **Recurrent Neural Networks (RNNs)**: Designed for sequential data
- **Transformers**: Revolutionary architecture for natural language processing

**Traditional AI Approaches**
- **Expert Systems**: Rule-based systems encoding human expertise
- **Search Algorithms**: Finding optimal solutions in problem spaces
- **Logic Programming**: Using formal logic for reasoning and inference
- **Genetic Algorithms**: Evolutionary approaches to optimization`
    },
    {
      title: "Practical Implementation: Building a Classification System",
      content: `Let's implement a complete machine learning pipeline that demonstrates key AI concepts:

\`\`\`python interactive
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Simulate student performance data
np.random.seed(42)
n_students = 1000

# Features: study_hours, previous_grades, attendance_rate, assignment_scores
study_hours = np.random.normal(5, 2, n_students)
previous_grades = np.random.normal(75, 15, n_students)
attendance_rate = np.random.beta(8, 2, n_students) * 100
assignment_scores = np.random.normal(80, 12, n_students)

# Create feature matrix
X = np.column_stack([study_hours, previous_grades, attendance_rate, assignment_scores])

# Generate target variable (pass/fail) based on features
performance_score = (
    0.3 * study_hours + 
    0.4 * (previous_grades / 100) + 
    0.2 * (attendance_rate / 100) + 
    0.1 * (assignment_scores / 100) + 
    np.random.normal(0, 0.1, n_students)
)
y = (performance_score > np.median(performance_score)).astype(int)

class StudentPerformancePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = ['Study Hours', 'Previous Grades', 'Attendance %', 'Assignment Scores']
        
    def train(self, X, y):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features for better performance
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate performance
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        print(f"Training Accuracy: {train_accuracy:.3f}")
        print(f"Testing Accuracy: {test_accuracy:.3f}")
        
        # Feature importance analysis
        importance = self.model.feature_importances_
        for i, feature in enumerate(self.feature_names):
            print(f"{feature}: {importance[i]:.3f}")
        
        return X_test_scaled, y_test, test_predictions
    
    def predict_student_success(self, study_hours, prev_grades, attendance, assignments):
        # Make prediction for a new student
        student_data = np.array([[study_hours, prev_grades, attendance, assignments]])
        student_scaled = self.scaler.transform(student_data)
        prediction = self.model.predict(student_scaled)[0]
        probability = self.model.predict_proba(student_scaled)[0]
        
        return prediction, probability

# Train and evaluate the model
predictor = StudentPerformancePredictor()
X_test, y_test, predictions = predictor.train(X, y)

# Test with sample students
print("\\nSample Predictions:")
sample_students = [
    (8, 85, 95, 90),  # High performer
    (3, 60, 70, 65),  # At-risk student
    (6, 75, 85, 80)   # Average student
]

for i, (hours, grades, attend, assign) in enumerate(sample_students):
    pred, prob = predictor.predict_student_success(hours, grades, attend, assign)
    result = "Success" if pred == 1 else "At Risk"
    confidence = max(prob) * 100
    print(f"Student {i+1}: {result} (Confidence: {confidence:.1f}%)")
\`\`\`

This example demonstrates several key AI concepts:
- **Data preprocessing** and feature engineering
- **Model training** and validation
- **Performance evaluation** with metrics
- **Feature importance** analysis for interpretability
- **Prediction** on new data

The Random Forest algorithm combines multiple decision trees to create a robust classifier that can handle complex patterns while providing insights into which features matter most for predictions.`
    },
    {
      title: "AI Applications Across Industries",
      content: `Artificial Intelligence has become a transformative force across virtually every industry, creating new possibilities and reshaping traditional approaches to problem-solving.

**Healthcare and Medicine**
- **Medical Imaging**: AI systems can detect cancer in radiology scans with accuracy matching or exceeding human specialists
- **Drug Discovery**: Machine learning accelerates the identification of promising drug compounds, reducing development time from decades to years
- **Personalized Medicine**: AI analyzes genetic data, medical history, and lifestyle factors to customize treatment plans
- **Electronic Health Records**: Natural language processing extracts insights from unstructured medical notes and reports

**Transportation and Logistics**
- **Autonomous Vehicles**: Deep learning enables cars to perceive their environment and make driving decisions
- **Route Optimization**: AI algorithms optimize delivery routes, reducing fuel consumption and improving efficiency
- **Traffic Management**: Smart traffic systems use real-time data to minimize congestion and improve safety
- **Predictive Maintenance**: Machine learning predicts when vehicles and infrastructure components need servicing

**Finance and Economics**
- **Algorithmic Trading**: AI systems execute trades at superhuman speeds based on market pattern analysis
- **Fraud Detection**: Machine learning identifies suspicious transactions and prevents financial crimes
- **Credit Scoring**: AI models assess loan default risk using diverse data sources beyond traditional credit history
- **Robo-Advisors**: Automated investment platforms provide personalized financial advice to millions of users

**Education and Learning**
- **Adaptive Learning**: AI personalizes educational content based on individual student progress and learning styles
- **Automated Grading**: Natural language processing enables automated assessment of essays and open-ended responses
- **Intelligent Tutoring**: AI tutors provide personalized instruction and feedback to students
- **Learning Analytics**: Machine learning analyzes student behavior to identify at-risk learners and optimize curricula

**Entertainment and Media**
- **Content Recommendation**: Streaming platforms use AI to suggest movies, music, and content based on user preferences
- **Game AI**: Intelligent non-player characters create more engaging and challenging gaming experiences
- **Content Creation**: AI generates music, art, and written content, augmenting human creativity
- **Real-time Translation**: Neural networks enable instant translation across languages in video calls and live events`
    },
    {
      title: "Ethical Considerations and Future Challenges",
      content: `As AI systems become more powerful and pervasive, addressing ethical implications and societal challenges becomes increasingly critical.

**Key Ethical Concerns**

*Bias and Fairness*
- AI systems can perpetuate or amplify existing societal biases present in training data
- Algorithmic discrimination in hiring, lending, and criminal justice systems
- Need for diverse development teams and inclusive design practices
- Importance of bias testing and fairness metrics

*Privacy and Surveillance*
- AI enables unprecedented data collection and analysis capabilities
- Facial recognition and behavioral tracking raise privacy concerns
- Balance between personalization benefits and privacy protection
- Need for transparent data usage policies and user consent

*Transparency and Explainability*
- Many AI systems operate as "black boxes" with opaque decision-making processes
- Critical need for explainable AI in high-stakes applications (healthcare, finance, legal)
- Trade-offs between model performance and interpretability
- Regulatory requirements for algorithmic transparency

**Future Challenges and Opportunities**

*Artificial General Intelligence (AGI)*
- Current AI is narrow and task-specific; AGI would match human cognitive abilities
- Timeline and feasibility remain subjects of intense debate
- Potential benefits: solving complex global challenges, scientific breakthroughs
- Risks: job displacement, loss of human agency, existential concerns

*Human-AI Collaboration*
- Focus on augmenting rather than replacing human capabilities
- AI as a tool to enhance human decision-making and creativity
- Need for new skills and educational approaches
- Importance of maintaining human oversight and control

*Regulatory and Governance Frameworks*
- Development of AI ethics guidelines and regulatory standards
- International cooperation on AI governance and safety
- Balancing innovation with responsible development
- Need for adaptive policies that keep pace with technological advancement

**Preparing for an AI-Driven Future**
- Continuous learning and skill development
- Understanding AI capabilities and limitations
- Engaging in discussions about AI's role in society
- Contributing to responsible AI development and deployment`
    }
  ]
}; 