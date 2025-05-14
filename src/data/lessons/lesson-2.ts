// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.
import { LessonData } from "./lesson-1";

export const lesson2: LessonData = {
  id: 2,
  title: "Getting Started with Programming",
  description: "Learn the basics of programming with Python, from variables to functions.",
  sections: [
    {
      title: "What is Programming?",
      content: `Programming is the process of creating instructions for computers to follow. Python is a great language for beginners.`
    },
    {
      title: "Variables and Data Types",
      content: `Variables store data that your program can use. Python supports several data types:

- **int**: Whole numbers (e.g., 5)
- **float**: Decimal numbers (e.g., 3.14)
- **str**: Text (e.g., "hello")
- **bool**: True or False values

Example:

\`\`\`python
name = "Alice"
age = 20
height = 1.75
is_student = True
print(name, age, height, is_student)
\`\`\`
    `},
    {
      title: "Operators",
      content: `Operators let you perform actions on variables and values:

- **Arithmetic**: +, -, *, /, %
- **Comparison**: ==, !=, >, <, >=, <=
- **Logical**: and, or, not

Example:

\`\`\`python
x = 10
y = 3
print(x + y)      # 13
print(x > y)      # True
print(x % y == 1) # True
\`\`\`
    `},
    {
      title: "Control Flow: if/else",
      content: `Control flow lets your program make decisions:

\`\`\`python
age = 18
if age >= 10:
    print("You are an adult.")
else:
    print("You are a minor.")
\`\`\`
    `},
    {
      title: "Loops",
      content: `Loops let you repeat actions:

- **for** loop: Repeat a block for each item in a sequence.
- **while** loop: Repeat as long as a condition is true.

Examples:

\`\`\`python
# For loop
for i in range(5):
    print(i)

# While loop
count = 0
while count < 3:
    print(count)
    count += 1
\`\`\`
    },
    {
      title: "Functions",
      content: \`Functions let you organize code into reusable blocks:
\`\`\`python
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
\`\`\`
    `},
    {
      title: "Try It Yourself!",
      content: `Write a function that checks if a number is even or odd:
\`\`\`python
def is_even(n):
    if n % 2 == 0:
        return True
    else:
        return False

print(is_even(4))  # True
print(is_even(7))  # False
\`\`\`
    `}
  ]
};
