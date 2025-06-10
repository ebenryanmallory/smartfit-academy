// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.

import type { LessonData } from "../types";

export const lesson1: LessonData = {
  id: 1,
  title: "Introduction to Artificial Intelligence",
  description: "Explore AI fundamentals, real-world applications, and get hands-on with basic programming concepts.",
  sections: [
    {
      title: "What is Artificial Intelligence?",
      content: `Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence. These tasks include:

- **Pattern Recognition**: Identifying objects in images or recognizing speech
- **Decision Making**: Choosing the best action based on available information  
- **Problem Solving**: Finding solutions to complex challenges
- **Learning**: Improving performance through experience

AI isn't just one technology—it's a collection of techniques that enable machines to simulate human cognitive abilities.`
    },
    {
      title: "Types of AI",
      content: `There are different categories of AI based on their capabilities:

**Narrow AI (Weak AI)**
- Designed for specific tasks (like image recognition or language translation)
- Most AI systems today fall into this category
- Examples: Spotify recommendations, Google Translate, Tesla's autopilot

**General AI (Strong AI)**
- Hypothetical AI that could perform any intellectual task a human can
- Doesn't exist yet, but is a long-term goal of AI research

**Superintelligence**
- AI that surpasses human intelligence in all domains
- Currently theoretical and the subject of much debate`
    },
    {
      title: "How AI Systems Learn",
      content: `AI systems learn through a process called **Machine Learning**, which involves training algorithms on data:

**Supervised Learning**
- AI learns from examples with correct answers
- Like studying for a test with answer keys
- Example: Training AI to recognize cats by showing it thousands of labeled cat photos

**Unsupervised Learning**
- AI finds patterns in data without being told what to look for
- Like discovering hidden connections in large datasets
- Example: Grouping customers by shopping behavior

**Reinforcement Learning**
- AI learns through trial and error, receiving rewards for good actions
- Like learning to play a game by practicing and getting feedback
- Example: AI playing chess and improving after each game`
    },
    {
      title: "Programming with AI Concepts",
      content: `Let's explore a simple example of how AI might make decisions using basic programming:

\`\`\`python interactive
import random

# Simple decision-making AI for a study recommendation system
def study_recommendation(hours_available, difficulty_preference, subject):
    # Simple rule-based decision making
    if hours_available < 1:
        return f"Quick review: Watch a 15-minute {subject} video"
    elif hours_available < 3:
        if difficulty_preference == "easy":
            return f"Practice basic {subject} problems for 1-2 hours"
        else:
            return f"Work on intermediate {subject} exercises"
    else:
        if difficulty_preference == "hard":
            return f"Tackle advanced {subject} projects for 3+ hours"
        else:
            return f"Deep dive into {subject} theory and practice"

# Test the recommendation system
print(study_recommendation(2, "easy", "math"))
print(study_recommendation(4, "hard", "physics"))
print(study_recommendation(0.5, "easy", "chemistry"))
\`\`\`

This simple system uses basic rules to make recommendations—real AI systems use much more sophisticated algorithms!`
    },
    {
      title: "AI in Your Daily Life",
      content: `AI is already integrated into many aspects of modern life:

**Social Media & Entertainment**
- News feed algorithms decide what posts you see
- Streaming services recommend movies and music
- Photo tagging automatically identifies people

**Communication**
- Autocorrect and predictive text on your phone
- Real-time language translation
- Spam email filtering

**Transportation**
- GPS navigation finding optimal routes
- Ride-sharing apps matching drivers and passengers
- Advanced driver assistance systems

**Shopping & Commerce**
- Product recommendations on e-commerce sites
- Price comparison and dynamic pricing
- Fraud detection for credit card transactions`
    },
    {
      title: "Career Paths in AI",
      content: `The AI field offers diverse career opportunities:

**Technical Roles**
- **Machine Learning Engineer**: Builds and deploys AI systems
- **Data Scientist**: Analyzes data to extract insights and build models
- **Research Scientist**: Develops new AI algorithms and techniques

**Applied Roles**
- **AI Product Manager**: Guides development of AI-powered products
- **AI Ethics Specialist**: Ensures AI systems are fair and responsible
- **AI Trainer**: Creates datasets and trains AI models

**Preparation Tips**
- **Math & Statistics**: Foundational for understanding AI algorithms
- **Programming**: Python is the most popular language for AI
- **Critical Thinking**: Essential for problem-solving and debugging
- **Communication**: Important for explaining AI concepts to others`
    },
    {
      title: "Getting Started with AI",
      content: `Ready to dive deeper into AI? Here's your roadmap:

**High School Preparation**
1. **Strengthen Math Skills**: Focus on algebra, statistics, and basic calculus
2. **Learn Programming**: Start with Python—it's beginner-friendly and widely used in AI
3. **Explore Online Resources**: Try platforms like Codecademy, Khan Academy, or Coursera
4. **Join Communities**: Participate in coding clubs or AI interest groups

**Hands-on Projects to Try**
- Create a simple chatbot using online tools
- Build a basic recommendation system
- Experiment with image recognition APIs
- Analyze data trends in spreadsheets

**College Planning**
- Consider majors in Computer Science, Data Science, or related fields
- Look for universities with strong AI research programs
- Participate in programming competitions and hackathons

The field of AI is rapidly growing, and there's never been a better time to get involved!`
    }
  ]
}; 