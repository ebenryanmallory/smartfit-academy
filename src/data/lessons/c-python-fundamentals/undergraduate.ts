// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.
import type { LessonData } from "../types";

export const lesson: LessonData = {
  id: "c-python-fundamentals",
  title: "Programming Fundamentals and Problem Solving",
  description: "Master essential programming concepts, data structures, and algorithmic thinking using Python for computer science applications.",
  sections: [
    {
      title: "Introduction to Programming and Computational Thinking",
      content: `Programming is the art and science of designing algorithms and implementing them as executable code. It combines mathematical precision with creative problem-solving to instruct computers in performing complex tasks.

**Core Programming Concepts:**
- **Algorithm**: A step-by-step procedure for solving a problem
- **Abstraction**: Hiding complex implementation details behind simple interfaces  
- **Decomposition**: Breaking large problems into smaller, manageable parts
- **Pattern Recognition**: Identifying similarities and reusable solutions

**Why Python for Computer Science?**
Python excels in readability and versatility, making it ideal for:
- Rapid prototyping and algorithm development
- Data analysis and scientific computing
- Web development and automation
- Machine learning and artificial intelligence
- System administration and scripting

Python's philosophy emphasizes code clarity and programmer productivity, following the principle that "code is read much more often than it is written."`
    },
    {
      title: "Variables, Data Types, and Memory Management",
      content: `Variables serve as symbolic names for memory locations that store data. Understanding how Python manages data types and memory is crucial for writing efficient programs.

**Primitive Data Types:**
- **int**: Arbitrary-precision integers (e.g., 42, -17, 1000000000000)
- **float**: IEEE 754 double-precision floating-point (e.g., 3.14159, 2.0e-5)
- **str**: Immutable Unicode text sequences (e.g., "Hello", 'Python')
- **bool**: Boolean values (True, False) - subclass of int
- **complex**: Complex numbers (e.g., 3+4j, 1.5-2.7j)
- **NoneType**: Represents absence of value (None)

**Variable Assignment and Object Identity:**
\`\`\`python interactive
# Variable assignment creates object references
student_name = "Alice Johnson"
student_id = 20230145
gpa = 3.85
is_honors = True
courses_enrolled = None  # Will be assigned later

# Python uses dynamic typing - variables can reference different types
print(f"Student: {student_name} (ID: {student_id})")
print(f"GPA: {gpa}, Honors: {is_honors}")
print(f"Type of student_id: {type(student_id)}")

# Object identity and equality
x = 1000
y = 1000
print(f"x == y: {x == y}")        # Value equality
print(f"x is y: {x is y}")        # Object identity (may be False for large numbers)

# Small integers are cached (singleton pattern)
a = 5
b = 5
print(f"a is b: {a is b}")        # True - same object in memory
\`\`\`

**String Operations and Methods:**
\`\`\`python interactive
course_name = "Computer Science"
course_code = "CS-101"

# String formatting and manipulation
welcome_msg = f"Welcome to {course_name} ({course_code})"
print(welcome_msg)

# Common string methods
print(f"Uppercase: {course_name.upper()}")
print(f"Length: {len(course_name)}")
print(f"Contains 'Science': {'Science' in course_name}")
print(f"Split words: {course_name.split()}")

# String slicing and indexing
print(f"First 8 characters: {course_name[:8]}")
print(f"Last word: {course_name[course_name.rfind(' ')+1:]}")
\`\`\`
    `},
    {
      title: "Data Structures: Lists, Tuples, and Dictionaries",
      content: `Python provides powerful built-in data structures that are essential for organizing and manipulating data efficiently.

**Lists - Mutable Sequences:**
\`\`\`python interactive
# Creating and manipulating lists
grades = [85, 92, 78, 96, 88]
students = ["Alice", "Bob", "Charlie", "Diana"]

# List operations and methods
grades.append(94)           # Add element to end
grades.insert(2, 90)        # Insert at specific index
print(f"Grades: {grades}")

# List comprehensions for efficient data processing
squared_grades = [grade**2 for grade in grades]
passing_grades = [grade for grade in grades if grade >= 80]
print(f"Squared: {squared_grades}")
print(f"Passing: {passing_grades}")

# Slicing and indexing
print(f"First three: {grades[:3]}")
print(f"Last two: {grades[-2:]}")
print(f"Every other: {grades[::2]}")
\`\`\`

**Tuples - Immutable Sequences:**
\`\`\`python interactive
# Tuples for structured data
student_record = ("Alice Johnson", 20230145, 3.85, "Computer Science")
coordinates = (10.5, 20.3)

# Tuple unpacking
name, student_id, gpa, major = student_record
print(f"Student: {name}, Major: {major}, GPA: {gpa}")

# Tuple as dictionary keys (immutable requirement)
grade_book = {
    ("CS101", "Spring2023"): 95,
    ("MATH201", "Fall2022"): 88,
    ("PHYS101", "Spring2023"): 92
}
\`\`\`

**Dictionaries - Key-Value Mappings:**
\`\`\`python interactive
# Student database using dictionaries
student_db = {
    "20230145": {
        "name": "Alice Johnson",
        "major": "Computer Science", 
        "gpa": 3.85,
        "courses": ["CS101", "MATH201", "PHYS101"]
    },
    "20230146": {
        "name": "Bob Smith",
        "major": "Mathematics",
        "gpa": 3.92,
        "courses": ["MATH301", "STAT200", "CS201"]
    }
}

# Dictionary operations
print(f"Alice's GPA: {student_db['20230145']['gpa']}")

# Safe access with get() method
bob_gpa = student_db.get("20230146", {}).get("gpa", "Not found")
print(f"Bob's GPA: {bob_gpa}")

# Dictionary comprehension
gpa_summary = {sid: data["gpa"] for sid, data in student_db.items()}
print(f"GPA Summary: {gpa_summary}")
\`\`\`
    `},
    {
      title: "Operators and Expressions",
      content: `Operators are symbols that perform computations on operands. Understanding operator precedence and associativity is crucial for writing correct expressions.

**Arithmetic Operators:**
\`\`\`python interactive
# Basic arithmetic operations
a, b = 17, 5
print(f"Addition: {a} + {b} = {a + b}")
print(f"Subtraction: {a} - {b} = {a - b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division: {a} / {b} = {a / b}")           # Float division
print(f"Floor Division: {a} // {b} = {a // b}")   # Integer division
print(f"Modulus: {a} % {b} = {a % b}")           # Remainder
print(f"Exponentiation: {a} ** {b} = {a ** b}")  # Power

# Operator precedence demonstration
result = 2 + 3 * 4 ** 2      # 2 + (3 * (4 ** 2)) = 50
print(f"2 + 3 * 4 ** 2 = {result}")
\`\`\`

**Comparison and Logical Operators:**
\`\`\`python interactive
# Comparison operators return boolean values
score = 85
threshold = 80

print(f"Score >= threshold: {score >= threshold}")
print(f"Score == 85: {score == 85}")
print(f"Score != 100: {score != 100}")

# Logical operators for complex conditions
has_prerequisites = True
gpa_requirement = 3.0
current_gpa = 3.5

eligible = has_prerequisites and current_gpa >= gpa_requirement
print(f"Course eligible: {eligible}")

# Chained comparisons (Pythonic feature)
grade = 87
letter_grade = "B" if 80 <= grade < 90 else "A" if grade >= 90 else "C"
print(f"Grade {grade} is: {letter_grade}")
\`\`\`
    `},
    {
      title: "Control Flow: Conditional Statements",
      content: `Conditional statements enable programs to execute different code paths based on boolean expressions, forming the foundation of decision-making in algorithms.

**Basic Conditional Structure:**
\`\`\`python interactive
def determine_grade_status(gpa, credit_hours):
    """Determine academic standing based on GPA and credit hours."""
    
    if gpa >= 3.5 and credit_hours >= 12:
        status = "Dean's List"
        eligible_honors = True
    elif gpa >= 3.0 and credit_hours >= 12:
        status = "Good Standing"
        eligible_honors = False
    elif gpa >= 2.0:
        status = "Academic Warning"
        eligible_honors = False
    else:
        status = "Academic Probation"
        eligible_honors = False
    
    return status, eligible_honors

# Test the function
student_gpa = 3.7
student_credits = 15
status, honors = determine_grade_status(student_gpa, student_credits)
print(f"GPA: {student_gpa}, Credits: {student_credits}")
print(f"Status: {status}, Honors Eligible: {honors}")
\`\`\`

**Advanced Conditional Patterns:**
\`\`\`python interactive
def calculate_tuition(credit_hours, residency, student_type):
    """Calculate tuition based on multiple factors."""
    
    # Base rates per credit hour
    base_rates = {
        ("undergraduate", "in_state"): 250,
        ("undergraduate", "out_state"): 650,
        ("graduate", "in_state"): 400,
        ("graduate", "out_state"): 850
    }
    
    rate_key = (student_type, residency)
    base_rate = base_rates.get(rate_key, 500)  # Default rate
    
    # Calculate base tuition
    base_tuition = credit_hours * base_rate
    
    # Apply discounts and fees
    if credit_hours >= 12:
        # Full-time student discount
        discount = 0.05 if student_type == "undergraduate" else 0.03
        base_tuition *= (1 - discount)
        
        # Add mandatory fees for full-time students
        base_tuition += 500
    
    # Honor roll discount
    if credit_hours >= 15 and student_type == "undergraduate":
        base_tuition *= 0.95  # 5% additional discount
    
    return round(base_tuition, 2)

# Test different scenarios
scenarios = [
    (12, "in_state", "undergraduate"),
    (9, "out_state", "graduate"),
    (16, "in_state", "undergraduate")
]

for credits, residency, student_type in scenarios:
    tuition = calculate_tuition(credits, residency, student_type)
    print(f"{student_type.title()} {residency.replace('_', '-')}: {credits} credits = \${tuition}")
\`\`\`
    `},
    {
      title: "Loops and Iteration",
      content: `Loops enable repetitive execution of code blocks, essential for processing collections and implementing algorithms efficiently.

**For Loops - Definite Iteration:**
\`\`\`python interactive
# Iterating over ranges
print("Counting from 1 to 5:")
for i in range(1, 6):
    print(f"Count: {i}")

# Iterating over data structures
students = ["Alice", "Bob", "Charlie", "Diana"]
for student in students:
    print(f"Hello, {student}!")

# Using enumerate for index access
print("\\nStudent roster:")
for index, student in enumerate(students, 1):
    print(f"{index}. {student}")

# Nested loops for matrix operations
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print("\\nMatrix elements:")
for row_idx, row in enumerate(matrix):
    for col_idx, element in enumerate(row):
        print(f"matrix[{row_idx}][{col_idx}] = {element}")
\`\`\`

**While Loops - Indefinite Iteration:**
\`\`\`python interactive
# User input validation simulation
def validate_grade_input(simulated_inputs):
    """Simulate grade input validation."""
    input_iter = iter(simulated_inputs)
    
    while True:
        try:
            grade_input = next(input_iter)
            grade = float(grade_input)
            
            if 0 <= grade <= 100:
                print(f"Valid grade entered: {grade}")
                return grade
            else:
                print(f"Grade {grade} is out of range. Please enter 0-100.")
        except (ValueError, StopIteration):
            print(f"Invalid input '{grade_input}'. Please enter a number.")
            return None

# Test with different inputs
test_inputs = ["105", "abc", "85.5"]
result = validate_grade_input(test_inputs)
print(f"Final result: {result}")
\`\`\`

**Advanced Loop Patterns:**
\`\`\`python interactive
# List comprehensions vs traditional loops
numbers = range(1, 11)

# Traditional approach
squares_traditional = []
for n in numbers:
    if n % 2 == 0:
        squares_traditional.append(n ** 2)

# List comprehension approach
squares_comprehension = [n ** 2 for n in numbers if n % 2 == 0]

print(f"Traditional: {squares_traditional}")
print(f"Comprehension: {squares_comprehension}")

# Dictionary comprehension for grade calculations
raw_scores = {"Alice": 85, "Bob": 92, "Charlie": 78, "Diana": 96}
letter_grades = {
    name: "A" if score >= 90 else "B" if score >= 80 else "C"
    for name, score in raw_scores.items()
}
print(f"Letter grades: {letter_grades}")
\`\`\`
    `},
    {
      title: "Functions and Modular Programming",
      content: `Functions are reusable code blocks that encapsulate specific functionality, promoting code organization, reusability, and maintainability.

**Function Definition and Parameters:**
\`\`\`python interactive
def calculate_gpa(grades, credit_hours):
    """Calculate GPA given grades and corresponding credit hours."""
    if len(grades) != len(credit_hours):
        raise ValueError("Grades and credit hours lists must have same length")
    
    total_grade_points = sum(grade * credits for grade, credits in zip(grades, credit_hours))
    total_credits = sum(credit_hours)
    
    return round(total_grade_points / total_credits, 2) if total_credits > 0 else 0.0

# Test the function
student_grades = [3.7, 4.0, 3.3, 3.8]
student_credits = [3, 4, 3, 2]
gpa = calculate_gpa(student_grades, student_credits)
print(f"Calculated GPA: {gpa}")
\`\`\`

**Default Parameters and Keyword Arguments:**
\`\`\`python interactive
def format_student_info(name, student_id, major="Undeclared", gpa=0.0, graduation_year=2027):
    """Format student information with default values."""
    return {
        "name": name,
        "id": student_id,
        "major": major,
        "gpa": gpa,
        "graduation_year": graduation_year,
        "academic_status": "Good Standing" if gpa >= 2.0 else "Probation"
    }

# Various ways to call the function
student1 = format_student_info("Alice Johnson", "20230145")
student2 = format_student_info("Bob Smith", "20230146", "Computer Science", 3.85)
student3 = format_student_info("Charlie Brown", "20230147", gpa=3.92, major="Mathematics")

print(f"Student 1: {student1}")
print(f"Student 2: {student2}")
print(f"Student 3: {student3}")
\`\`\`

**Advanced Function Features:**
\`\`\`python interactive
def process_course_data(*args, **kwargs):
    """Demonstrate variable arguments and keyword arguments."""
    print(f"Positional arguments: {args}")
    print(f"Keyword arguments: {kwargs}")
    
    # Process course codes (positional args)
    courses = list(args)
    
    # Process additional info (keyword args)
    semester = kwargs.get("semester", "Unknown")
    instructor = kwargs.get("instructor", "TBA")
    
    return {
        "courses": courses,
        "semester": semester,
        "instructor": instructor,
        "total_courses": len(courses)
    }

# Call with various argument patterns
result = process_course_data("CS101", "MATH201", "PHYS101", 
                           semester="Fall 2023", 
                           instructor="Dr. Smith")
print(f"Course data: {result}")
\`\`\`

**Lambda Functions and Higher-Order Functions:**
\`\`\`python interactive
# Lambda functions for simple operations
students_data = [
    {"name": "Alice", "gpa": 3.8, "credits": 15},
    {"name": "Bob", "gpa": 3.2, "credits": 12},
    {"name": "Charlie", "gpa": 3.9, "credits": 18},
    {"name": "Diana", "gpa": 2.8, "credits": 10}
]

# Using built-in higher-order functions
honor_students = list(filter(lambda s: s["gpa"] >= 3.5, students_data))
gpa_list = list(map(lambda s: s["gpa"], students_data))
total_credits = sum(map(lambda s: s["credits"], students_data))

print(f"Honor students: {[s['name'] for s in honor_students]}")
print(f"All GPAs: {gpa_list}")
print(f"Total credits: {total_credits}")

# Sorting with custom key functions
sorted_by_gpa = sorted(students_data, key=lambda s: s["gpa"], reverse=True)
print(f"Students by GPA: {[s['name'] for s in sorted_by_gpa]}")
\`\`\`
    `},
    {
      title: "Practical Application: Student Management System",
      content: `Let's integrate all the concepts we've learned to build a comprehensive student management system that demonstrates real-world programming applications.

**Complete Student Management Implementation:**
\`\`\`python interactive
class StudentManager:
    """A comprehensive system for managing student records and academic data."""
    
    def __init__(self):
        self.students = {}
        self.courses = {}
        self.grade_scale = {
            'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7,
            'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D+': 1.3, 'D': 1.0, 'F': 0.0
        }
    
    def add_student(self, student_id, name, major="Undeclared"):
        """Add a new student to the system."""
        if student_id in self.students:
            print(f"Student {student_id} already exists!")
            return False
        
        self.students[student_id] = {
            'name': name,
            'major': major,
            'courses': {},
            'total_credits': 0,
            'gpa': 0.0
        }
        print(f"Added student: {name} ({student_id})")
        return True
    
    def add_course(self, course_code, course_name, credits):
        """Add a course to the system."""
        self.courses[course_code] = {
            'name': course_name,
            'credits': credits
        }
        print(f"Added course: {course_name} ({course_code}) - {credits} credits")
    
    def enroll_student(self, student_id, course_code, grade=None):
        """Enroll a student in a course and optionally assign a grade."""
        if student_id not in self.students:
            print(f"Student {student_id} not found!")
            return False
        
        if course_code not in self.courses:
            print(f"Course {course_code} not found!")
            return False
        
        self.students[student_id]['courses'][course_code] = {
            'grade': grade,
            'credits': self.courses[course_code]['credits']
        }
        
        if grade:
            self._update_gpa(student_id)
        
        print(f"Enrolled student {student_id} in {course_code}")
        return True
    
    def assign_grade(self, student_id, course_code, letter_grade):
        """Assign a grade to a student for a specific course."""
        if (student_id not in self.students or 
            course_code not in self.students[student_id]['courses']):
            print("Student or course enrollment not found!")
            return False
        
        if letter_grade not in self.grade_scale:
            print(f"Invalid grade: {letter_grade}")
            return False
        
        self.students[student_id]['courses'][course_code]['grade'] = letter_grade
        self._update_gpa(student_id)
        print(f"Assigned grade {letter_grade} to student {student_id} for {course_code}")
        return True
    
    def _update_gpa(self, student_id):
        """Recalculate and update student's GPA."""
        student = self.students[student_id]
        total_points = 0
        total_credits = 0
        
        for course_data in student['courses'].values():
            if course_data['grade'] and course_data['grade'] in self.grade_scale:
                credits = course_data['credits']
                points = self.grade_scale[course_data['grade']] * credits
                total_points += points
                total_credits += credits
        
        student['total_credits'] = total_credits
        student['gpa'] = round(total_points / total_credits, 2) if total_credits > 0 else 0.0
    
    def get_student_transcript(self, student_id):
        """Generate a comprehensive transcript for a student."""
        if student_id not in self.students:
            return "Student not found!"
        
        student = self.students[student_id]
        transcript = f"""
        OFFICIAL TRANSCRIPT
        ==================
        Student: {student['name']} ({student_id})
        Major: {student['major']}
        Total Credits: {student['total_credits']}
        Cumulative GPA: {student['gpa']}
        
        Course History:
        """
        
        for course_code, course_data in student['courses'].items():
            course_name = self.courses[course_code]['name']
            grade = course_data['grade'] or "In Progress"
            credits = course_data['credits']
            transcript += f"        {course_code}: {course_name} ({credits} cr) - {grade}\\n"
        
        # Academic status
        gpa = student['gpa']
        if gpa >= 3.5:
            status = "Dean's List"
        elif gpa >= 3.0:
            status = "Good Standing"
        elif gpa >= 2.0:
            status = "Academic Warning"
        else:
            status = "Academic Probation"
        
        transcript += f"\\n        Academic Status: {status}"
        return transcript
    
    def generate_class_report(self, course_code):
        """Generate a report for all students in a specific course."""
        if course_code not in self.courses:
            return "Course not found!"
        
        enrolled_students = []
        for student_id, student_data in self.students.items():
            if course_code in student_data['courses']:
                grade = student_data['courses'][course_code]['grade']
                enrolled_students.append({
                    'id': student_id,
                    'name': student_data['name'],
                    'grade': grade or "No Grade"
                })
        
        course_name = self.courses[course_code]['name']
        report = f"""
        CLASS REPORT: {course_name} ({course_code})
        =======================================
        Total Enrolled: {len(enrolled_students)}
        
        Student List:
        """
        
        for student in sorted(enrolled_students, key=lambda x: x['name']):
            report += f"        {student['id']}: {student['name']} - {student['grade']}\\n"
        
        return report

# Demonstrate the system
print("=== Student Management System Demo ===")
manager = StudentManager()

# Add courses
manager.add_course("CS101", "Introduction to Computer Science", 4)
manager.add_course("MATH201", "Calculus I", 4)
manager.add_course("PHYS101", "General Physics", 3)

# Add students
manager.add_student("20230145", "Alice Johnson", "Computer Science")
manager.add_student("20230146", "Bob Smith", "Mathematics")

# Enroll students and assign grades
manager.enroll_student("20230145", "CS101")
manager.enroll_student("20230145", "MATH201")
manager.assign_grade("20230145", "CS101", "A")
manager.assign_grade("20230145", "MATH201", "B+")

manager.enroll_student("20230146", "MATH201")
manager.assign_grade("20230146", "MATH201", "A-")

# Generate reports
print(manager.get_student_transcript("20230145"))
print(manager.generate_class_report("MATH201"))
\`\`\`

This comprehensive example demonstrates:
- **Object-oriented design** with classes and methods
- **Data structure usage** with dictionaries and lists
- **Control flow** with conditional statements and loops
- **Function design** with parameters, return values, and error handling
- **String formatting** and report generation
- **Real-world application** of programming concepts

🎯 **Challenge**: Extend this system by adding features like course prerequisites, semester tracking, or grade point calculations for different academic standings.`
    }
  ]
};
