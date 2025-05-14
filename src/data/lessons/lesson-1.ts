// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.

export interface LessonSection {
  title: string;
  content: string;
}

export interface LessonData {
  id: number;
  title: string;
  description: string;
  sections: LessonSection[];
}

export const lesson1: LessonData = {
  id: 1,
  title: "Introduction to AI",
  description: "Learn the fundamentals of Artificial Intelligence and its impact on our world.",
  sections: [
    {
      title: "What is Artificial Intelligence?",
      content: `Artificial Intelligence (AI) is the simulation of human intelligence by machines that are programmed to think and learn like humans.`
    },
    {
      title: "Key Concepts",
      content: `1. **Machine Learning**: A subset of AI that enables systems to learn and improve from experience
2. **Neural Networks**: Computing systems inspired by the human brain
3. **Deep Learning**: A type of machine learning based on artificial neural networks`
    },
    {
      title: "A Simple Neural Network Example",
      content: `Here's a basic example of a neural network using Python and PyTorch:

\`\`\`python
import torch
import torch.nn as nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 1)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Create and test the network
model = SimpleNeuralNetwork()
input_data = torch.randn(1, 10)
output = model(input_data)
print(f"Input shape: {input_data.shape}")
print(f"Output shape: {output.shape}")
\`\`\`
`
    },
    {
      title: "Try It Yourself!",
      content: `Let's create a simple Python function to calculate the Fibonacci sequence. You can edit and run this code in the interactive playground below:

\`\`\`python interactive
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the function
for i in range(10):
    print(f"Fibonacci({i}) = {fibonacci(i)}")
\`\`\`
`
    },
    {
      title: "Why AI Matters",
      content: `AI is transforming industries and creating new opportunities. From healthcare to transportation, AI is making our lives better and more efficient.`
    },
    {
      title: "Getting Started",
      content: `To begin your AI journey, you'll need to understand:
- Basic programming concepts
- Mathematics fundamentals
- Problem-solving skills

Ready to dive deeper? Sign in to save your progress and access more lessons!`
    },
    {
      title: "AI in Everyday Life",
      content: `Artificial Intelligence is already part of our daily routines. One simple example is a rule-based chatbot, which can answer basic questions based on predefined rules.

Here's a simple Python example of a rule-based chatbot:

\`\`\`python
def chatbot(input_text):
    if "hello" in input_text.lower():
        return "Hello! How can I help you today?"
    elif "weather" in input_text.lower():
        return "I can't check the weather yet, but it's always a good day to learn AI!"
    else:
        return "I'm still learning. Can you ask something else?"

# Try the chatbot
print(chatbot("hello"))
print(chatbot("What's the weather?"))
print(chatbot("Tell me a joke"))
\`\`\`

This example shows how simple logic can create interactive programs. More advanced AI chatbots use machine learning to understand and generate responses.`
    }
  ]
};
