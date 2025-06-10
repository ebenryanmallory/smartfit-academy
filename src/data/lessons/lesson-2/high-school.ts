// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.

import type { LessonData } from "../types";

export const lesson2: LessonData = {
  id: 2,
  title: "Programming Fundamentals with Python",
  description: "Master the core concepts of programming including variables, control structures, and functions through hands-on Python examples.",
  sections: [
    {
      title: "What is Programming?",
      content: `Programming is the process of designing and building executable computer instructions to accomplish specific tasks. It involves:

- **Problem Solving**: Breaking down complex problems into smaller, manageable steps
- **Logical Thinking**: Creating step-by-step solutions using algorithms
- **Code Implementation**: Translating algorithms into a programming language

Python is an excellent first language because of its readable syntax and versatility. It's used in web development, data science, artificial intelligence, automation, and more.`
    },
    {
      title: "Variables and Data Types",
      content: `Variables are containers that store data values. Python is dynamically typed, meaning you don't need to declare variable types explicitly.

**Core Data Types:**
- **int**: Integers (whole numbers)
- **float**: Floating-point numbers (decimals)
- **str**: Strings (text)
- **bool**: Boolean values (True/False)
- **list**: Ordered collections of items
- **dict**: Key-value pairs

\`\`\`python interactive
# Variable assignment and type checking
student_name = "Alex Johnson"
student_id = 12345
gpa = 3.85
is_honor_student = True
courses = ["Math", "Physics", "Computer Science"]
student_info = {"name": student_name, "id": student_id, "gpa": gpa}

print(f"Student: {student_name}")
print(f"Type of student_id: {type(student_id)}")
print(f"Courses: {courses}")
print(f"Honor student: {is_honor_student}")
\`\`\`

**Naming Conventions:**
- Use descriptive names: \`student_grade\` instead of \`sg\`
- Use snake_case for variables and functions
- Constants should be UPPER_CASE`
    },
    {
      title: "Operators and Expressions",
      content: `Operators perform operations on variables and values. Understanding operator precedence is crucial for writing correct expressions.

**Arithmetic Operators:**
\`\`\`python interactive
# Basic arithmetic
x, y = 15, 4
print(f"Addition: {x} + {y} = {x + y}")
print(f"Subtraction: {x} - {y} = {x - y}")
print(f"Multiplication: {x} * {y} = {x * y}")
print(f"Division: {x} / {y} = {x / y}")
print(f"Floor Division: {x} // {y} = {x // y}")
print(f"Modulus: {x} % {y} = {x % y}")
print(f"Exponentiation: {x} ** 2 = {x ** 2}")
\`\`\`

**Comparison and Logical Operators:**
\`\`\`python interactive
age = 17
has_license = True
has_car = False

# Comparison operators
print(f"Age >= 16: {age >= 16}")
print(f"Age == 18: {age == 18}")

# Logical operators
can_drive = age >= 16 and has_license
needs_ride = not has_car or not has_license
print(f"Can drive: {can_drive}")
print(f"Needs ride: {needs_ride}")
\`\`\``
    },
    {
      title: "Control Flow: Conditional Statements",
      content: `Conditional statements allow programs to make decisions based on different conditions.

\`\`\`python interactive
def determine_grade_level(age):
    if age < 14:
        return "Middle School"
    elif age < 18:
        return "High School"
    elif age < 22:
        return "College"
    else:
        return "Adult"

def calculate_letter_grade(percentage):
    if percentage >= 90:
        return "A"
    elif percentage >= 80:
        return "B"
    elif percentage >= 70:
        return "C"
    elif percentage >= 60:
        return "D"
    else:
        return "F"

# Test the functions
student_age = 16
test_score = 87

print(f"Grade level: {determine_grade_level(student_age)}")
print(f"Letter grade: {calculate_letter_grade(test_score)}")

# Nested conditionals for more complex logic
if student_age >= 16:
    if test_score >= 80:
        print("Eligible for advanced placement!")
    else:
        print("Consider tutoring to improve grades.")
else:
    print("Focus on building foundational skills.")
\`\`\``
    },
    {
      title: "Loops: Iteration and Repetition",
      content: `Loops allow you to execute code repeatedly, which is essential for processing collections of data.

**For Loops - Definite Iteration:**
\`\`\`python interactive
# Loop through a range
print("Countdown:")
for i in range(10, 0, -1):
    print(i)
print("Launch! 🚀")

# Loop through lists
subjects = ["Math", "Science", "History", "English"]
for subject in subjects:
    print(f"Studying {subject}")

# Loop with enumerate for index access
for index, subject in enumerate(subjects):
    print(f"{index + 1}. {subject}")
\`\`\`

**While Loops - Indefinite Iteration:**
\`\`\`python interactive
# Simulate a simple game score system
score = 0
level = 1

while score < 100:
    points_earned = level * 10
    score += points_earned
    print(f"Level {level}: Earned {points_earned} points. Total: {score}")
    level += 1

print(f"Congratulations! Final score: {score}")
\`\`\`

**Loop Control:**
\`\`\`python interactive
# Using break and continue
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print("Even numbers only:")
for num in numbers:
    if num % 2 != 0:
        continue  # Skip odd numbers
    if num > 6:
        break     # Stop at 6
    print(num)
\`\`\``
    },
    {
      title: "Functions: Code Organization and Reusability",
      content: `Functions are reusable blocks of code that perform specific tasks. They promote code organization and reduce repetition.

\`\`\`python interactive
def calculate_gpa(grades):
    """Calculate GPA from a list of numerical grades."""
    if not grades:
        return 0.0
    
    total_points = sum(grades)
    return total_points / len(grades)

def grade_to_points(letter_grade):
    """Convert letter grade to GPA points."""
    grade_points = {
        'A': 4.0, 'A-': 3.7,
        'B+': 3.3, 'B': 3.0, 'B-': 2.7,
        'C+': 2.3, 'C': 2.0, 'C-': 1.7,
        'D+': 1.3, 'D': 1.0, 'F': 0.0
    }
    return grade_points.get(letter_grade.upper(), 0.0)

def analyze_academic_performance(student_grades):
    """Comprehensive analysis of student performance."""
    points = [grade_to_points(grade) for grade in student_grades]
    gpa = calculate_gpa(points)
    
    if gpa >= 3.5:
        status = "Honor Roll"
    elif gpa >= 3.0:
        status = "Good Standing"
    elif gpa >= 2.0:
        status = "Probation Warning"
    else:
        status = "Academic Probation"
    
    return {
        'gpa': round(gpa, 2),
        'status': status,
        'total_courses': len(student_grades)
    }

# Example usage
semester_grades = ['A', 'B+', 'A-', 'B', 'C+']
results = analyze_academic_performance(semester_grades)

print(f"Semester Results:")
print(f"GPA: {results['gpa']}")
print(f"Status: {results['status']}")
print(f"Courses Completed: {results['total_courses']}")
\`\`\``
    },
    {
      title: "Practical Application: Student Management System",
      content: `Let's build a simple student management system that combines all the concepts we've learned:

\`\`\`python interactive
class StudentTracker:
    def __init__(self):
        self.students = {}
    
    def add_student(self, student_id, name):
        """Add a new student to the system."""
        self.students[student_id] = {
            'name': name,
            'grades': [],
            'attendance': 0
        }
        print(f"Added student: {name} (ID: {student_id})")
    
    def add_grade(self, student_id, grade):
        """Add a grade for a student."""
        if student_id in self.students:
            self.students[student_id]['grades'].append(grade)
            print(f"Grade {grade} added for student {student_id}")
        else:
            print("Student not found!")
    
    def calculate_average(self, student_id):
        """Calculate the average grade for a student."""
        if student_id in self.students:
            grades = self.students[student_id]['grades']
            if grades:
                return sum(grades) / len(grades)
        return 0
    
    def generate_report(self, student_id):
        """Generate a comprehensive report for a student."""
        if student_id not in self.students:
            return "Student not found!"
        
        student = self.students[student_id]
        average = self.calculate_average(student_id)
        
        report = f"""
        Student Report
        ==============
        Name: {student['name']}
        ID: {student_id}
        Grades: {student['grades']}
        Average: {average:.2f}
        Grade Count: {len(student['grades'])}
        """
        
        if average >= 90:
            report += "Performance: Excellent!"
        elif average >= 80:
            report += "Performance: Good"
        elif average >= 70:
            report += "Performance: Satisfactory"
        else:
            report += "Performance: Needs Improvement"
        
        return report

# Demo the system
tracker = StudentTracker()
tracker.add_student(12345, "Emma Wilson")
tracker.add_student(12346, "David Chen")

tracker.add_grade(12345, 92)
tracker.add_grade(12345, 88)
tracker.add_grade(12345, 95)

print(tracker.generate_report(12345))
\`\`\`

This example demonstrates how programming concepts work together to solve real-world problems!`
    }
  ]
}; 