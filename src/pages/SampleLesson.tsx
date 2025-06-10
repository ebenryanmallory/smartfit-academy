import { LessonViewer } from "../components/LessonViewer";

const lessonTitle = "Lesson 2: Python Basics";
const lessonDescription = "Learn the basics of programming with Python: variables, data types, operators, control flow, loops, and functions.";

import Footer from "../components/Footer";

const lessonContent = [
  "## Variables and Data Types",
  "Variables are used to store data. Python supports several data types, such as `int`, `float`, `str`, and `bool`.",
  "",
  "```python",
  "x = 5      # int",
  "y = 3.14   # float",
  "name = \"Alice\"  # str",
  "is_active = True  # bool",
  "```",
  "",
  "## Operators",
  "Python supports arithmetic, comparison, and logical operators.",
  "",
  "```python",
  "a = 10",
  "b = 3",
  "sum = a + b      # Addition",
  "is_equal = a == b  # Comparison",
  "is_both = (a > 0) and (b > 0)  # Logical",
  "```",
  "",
  "## Control Flow (if/else)",
  "Use `if`, `elif`, and `else` to control program flow.",
  "",
  "```python",
  "x = 7",
  "if x > 5:",
  "    print(\"x is greater than 5\")",
  "elif x == 5:",
  "    print(\"x is 5\")",
  "else:",
  "    print(\"x is less than 5\")",
  "```",
  "",
  "## Loops",
  "Loops help repeat actions. Python has `for` and `while` loops.",
  "",
  "```python",
  "for i in range(3):",
  "    print(i)",
  "",
  "count = 0",
  "while count < 3:",
  "    print(count)",
  "    count += 1",
  "```",
  "",
  "## Functions",
  "Functions group reusable code blocks.",
  "",
  "```python",
  "def greet(name):",
  "    print(f\"Hello, {name}!\")",
  "",
  "greet(\"Alice\")",
  "```"
].join("\n");

const SampleLesson = () => {
  return (
    <div className="content-container mx-auto p-8 space-y-12">
      <header className="space-y-2">
        <h1 className="text-4xl font-bold text-foreground">{lessonTitle}</h1>
        <p className="text-muted-foreground">{lessonDescription}</p>
      </header>
      <section>
        <LessonViewer title={lessonTitle} description={lessonDescription} content={lessonContent} />
      </section>
      <Footer />
    </div>
  );
};

export default SampleLesson;
