import { useParams, useNavigate } from 'react-router-dom';
import { LessonViewer } from "@/components/LessonViewer";
import { Button } from "@/components/ui/button";

interface Lesson {
  title: string;
  description: string;
  content: string;
}

const lessons: Record<string, Lesson> = {
  "1": {
    title: "Introduction to AI",
    description: "Learn the fundamentals of Artificial Intelligence and its impact on our world.",
    content: `
# What is Artificial Intelligence?

Artificial Intelligence (AI) is the simulation of human intelligence by machines that are programmed to think and learn like humans.

## Key Concepts

1. **Machine Learning**: A subset of AI that enables systems to learn and improve from experience
2. **Neural Networks**: Computing systems inspired by the human brain
3. **Deep Learning**: A type of machine learning based on artificial neural networks

## A Simple Neural Network Example

Here's a basic example of a neural network using Python and PyTorch:

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

## Try It Yourself!

Let's create a simple Python function to calculate the Fibonacci sequence. You can edit and run this code in the interactive playground below:

\`\`\`python interactive
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the function
for i in range(10):
    print(f"Fibonacci({i}) = {fibonacci(i)}")
\`\`\`

## Why AI Matters

AI is transforming industries and creating new opportunities. From healthcare to transportation, AI is making our lives better and more efficient.

## Getting Started

To begin your AI journey, you'll need to understand:
- Basic programming concepts
- Mathematics fundamentals
- Problem-solving skills

Ready to dive deeper? Sign in to save your progress and access more lessons!
    `
  },
  "2": {
    title: "Getting Started with Programming",
    description: "Learn the basics of programming with Python, from variables to functions.",
    content: `
# Getting Started with Programming

Welcome to your first programming lesson! We'll be using Python, a beginner-friendly programming language that's perfect for learning the fundamentals.

## What is Programming?

Programming is the process of creating a set of instructions that tell a computer how to perform a task. Think of it like writing a recipe - you need to be precise and clear in your instructions!

## Your First Program

Let's start with a simple "Hello, World!" program:

\`\`\`python
print("Hello, World!")
\`\`\`

This program uses the \`print()\` function to display text on the screen. Try running it in the playground below:

\`\`\`python interactive
print("Hello, World!")
\`\`\`

## Variables and Data Types

Variables are like containers that store information. In Python, we have several basic data types:

1. **Strings**: Text (like "Hello")
2. **Numbers**: Integers (1, 2, 3) and Floats (1.5, 2.7)
3. **Booleans**: True or False

Here's how to use variables:

\`\`\`python
name = "Alice"  # String
age = 25        # Integer
height = 1.75   # Float
is_student = True  # Boolean

print(f"Name: {name}")
print(f"Age: {age}")
print(f"Height: {height}")
print(f"Is student: {is_student}")
\`\`\`

Try creating your own variables in the playground:

\`\`\`python interactive
# Create your own variables here
name = "Your Name"
age = 20
favorite_number = 42

# Print them using f-strings
print(f"Hello, my name is {name}")
print(f"I am {age} years old")
print(f"My favorite number is {favorite_number}")
\`\`\`

## Basic Operations

Python can perform various operations on numbers:

\`\`\`python
# Addition
sum = 5 + 3
print(f"5 + 3 = {sum}")

# Subtraction
difference = 10 - 4
print(f"10 - 4 = {difference}")

# Multiplication
product = 6 * 7
print(f"6 * 7 = {product}")

# Division
quotient = 20 / 4
print(f"20 / 4 = {quotient}")
\`\`\`

Try some calculations in the playground:

\`\`\`python interactive
# Try your own calculations
num1 = 15
num2 = 3

# Addition
print(f"{num1} + {num2} = {num1 + num2}")

# Subtraction
print(f"{num1} - {num2} = {num1 - num2}")

# Multiplication
print(f"{num1} * {num2} = {num1 * num2}")

# Division
print(f"{num1} / {num2} = {num1 / num2}")
\`\`\`

## Functions

Functions are reusable blocks of code that perform specific tasks. Here's a simple function:

\`\`\`python
def greet(name):
    return f"Hello, {name}!"

# Using the function
message = greet("Bob")
print(message)
\`\`\`

Try creating your own function in the playground:

\`\`\`python interactive
def calculate_area(length, width):
    area = length * width
    return f"The area is {area} square units"

# Try the function with different values
print(calculate_area(5, 3))
print(calculate_area(10, 2))
\`\`\`

## Next Steps

You've learned the basics of:
- Printing text
- Using variables
- Basic operations
- Creating functions

Ready to learn more? Sign in to:
- Access more programming lessons
- Get personalized recommendations
- Track your progress
- Join our learning community
    `
  }
};

export default function LessonPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const lesson = id ? lessons[id] : undefined;

  if (!lesson) {
    return (
      <div className="container mx-auto py-12 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl font-bold mb-4">Lesson Not Found</h1>
          <p className="text-muted-foreground mb-8">
            The lesson you're looking for doesn't exist or isn't available yet.
          </p>
          <Button onClick={() => navigate('/lessons')}>
            Back to Lessons
          </Button>
        </div>
      </div>
    );
  }

  return <LessonViewer {...lesson} />;
} 