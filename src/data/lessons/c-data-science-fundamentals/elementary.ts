// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.

import type { LessonData } from "../types";

export const lesson: LessonData = {
  id: "c-data-science-fundamentals",
  title: "Data Detective Adventures - For Young Explorers",
  description: "Become a data detective and learn how to collect, organize, and discover amazing secrets hidden in information all around us!",
  sections: [
    {
      title: "What is Data?",
      content: `Data is everywhere around us! It's like collecting clues to solve a mystery.

Think about it:
- How many pets your friends have
- What your favorite ice cream flavors are
- How tall you are compared to your classmates
- What time you wake up each day

All of these are examples of DATA! Data is just information we collect about the world around us. When we have lots of data, we can discover cool patterns and learn amazing things! 🕵️‍♀️🔍`
    },
    {
      title: "Becoming a Data Detective",
      content: `Data detectives (also called Data Scientists) are like super cool investigators who solve mysteries using information!

**What do Data Detectives do?**
- 🔎 **Collect clues (data)** - Gather lots of information
- 📊 **Organize evidence** - Put information in neat groups and lists  
- 🧩 **Find patterns** - Look for things that repeat or connect
- 💡 **Solve mysteries** - Answer questions using what they found
- 📈 **Share discoveries** - Tell others about their cool findings

**Real Data Detective Work:**
- Helping zoos take better care of animals
- Finding out which playgrounds kids like best
- Discovering what makes plants grow faster
- Learning how to keep people healthy and happy

You can be a data detective too! Let's start our first investigation! 🎒🔍`
    },
    {
      title: "Our First Data Collection Adventure",
      content: `Let's collect data about our classroom and see what we can discover!

\`\`\`python interactive
# Let's collect data about favorite school subjects!
favorite_subjects = ["math", "art", "science", "reading", "art", "math", "science", "art", "music", "math"]

print("🎒 Our Class Survey Data:")
print("Favorite subjects:", favorite_subjects)

# Count how many times each subject appears
subject_count = {}
for subject in favorite_subjects:
    if subject in subject_count:
        subject_count[subject] = subject_count[subject] + 1
    else:
        subject_count[subject] = 1

print("\\n📊 What we discovered:")
for subject, count in subject_count.items():
    print(f"📚 {subject}: {count} students")

# Find the most popular subject
most_popular = ""
highest_count = 0

for subject, count in subject_count.items():
    if count > highest_count:
        highest_count = count
        most_popular = subject

print(f"\\n🏆 Most popular subject: {most_popular} with {highest_count} votes!")
print("\\n🎉 Great detective work! We learned something new about our class!")
\`\`\`

Try changing the favorite subjects list to see what happens with different data!`
    },
    {
      title: "Making Data Charts and Graphs",
      content: `Pictures help us see patterns in data! Let's create simple charts to visualize our discoveries.

\`\`\`python interactive
# Let's track how many books students read each month
students = ["Alice", "Bob", "Charlie", "Diana", "Eva"]
books_read = [3, 7, 5, 9, 4]

print("📚 Books Read This Month:")
print("Student Name -> Books Read")

# Create a simple text chart
for i in range(len(students)):
    student = students[i]
    books = books_read[i]
    
    # Create a visual bar using stars
    bar = "⭐" * books
    print(f"{student}: {bar} ({books} books)")

# Find interesting facts
total_books = sum(books_read)
average_books = total_books / len(books_read)
best_reader_index = books_read.index(max(books_read))
best_reader = students[best_reader_index]

print(f"\\n📈 Data Detective Discoveries:")
print(f"📖 Total books read by everyone: {total_books}")
print(f"📊 Average books per student: {average_books:.1f}")
print(f"🏆 Best reader: {best_reader} with {max(books_read)} books!")

# Let's sort from most to least books read
student_book_pairs = list(zip(students, books_read))
sorted_pairs = sorted(student_book_pairs, key=lambda x: x[1], reverse=True)

print(f"\\n🥇 Reading Championship Leaderboard:")
for i, (student, books) in enumerate(sorted_pairs):
    medal = ["🥇", "🥈", "🥉", "🏅", "🏅"][i]
    print(f"{medal} {student}: {books} books")
\`\`\`

Try adding your name and book count to see where you rank!`
    },
    {
      title: "Weather Data Detective",
      content: `Let's be weather detectives and analyze temperature data to predict what to wear!

\`\`\`python interactive
# Temperature data for a week (in Fahrenheit)
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
temperatures = [75, 68, 72, 80, 77, 85, 82]

print("🌤️ Weekly Weather Report:")
for i in range(len(days)):
    day = days[i]
    temp = temperatures[i]
    
    # Give clothing suggestions based on temperature
    if temp >= 80:
        outfit = "🩳 Shorts and t-shirt!"
    elif temp >= 70:
        outfit = "👕 T-shirt and pants"
    elif temp >= 60:
        outfit = "🧥 Light jacket"
    else:
        outfit = "🧥 Warm jacket"
    
    print(f"{day}: {temp}°F - {outfit}")

# Weather detective analysis
hottest_temp = max(temperatures)
coldest_temp = min(temperatures)
average_temp = sum(temperatures) / len(temperatures)

hottest_day = days[temperatures.index(hottest_temp)]
coldest_day = days[temperatures.index(coldest_temp)]

print(f"\\n🔍 Weather Detective Report:")
print(f"🌡️ Hottest day: {hottest_day} at {hottest_temp}°F")
print(f"❄️ Coldest day: {coldest_day} at {coldest_temp}°F") 
print(f"📊 Average temperature: {average_temp:.1f}°F")

# Predict weekend activities
weekend_temps = [temperatures[5], temperatures[6]]  # Saturday, Sunday
weekend_average = sum(weekend_temps) / len(weekend_temps)

print(f"\\n🎯 Weekend Prediction:")
if weekend_average >= 80:
    print("🏖️ Great weather for swimming or going to the beach!")
elif weekend_average >= 70:
    print("🚴‍♀️ Perfect for bike riding or playing outside!")
else:
    print("🎮 Good day for indoor activities or museums!")
\`\`\`

Try changing the temperatures to see different outfit suggestions!`
    },
    {
      title: "Pet Data Detective Agency",
      content: `Let's help a pet store owner learn about their animals using data detective skills!

\`\`\`python interactive
# Pet store data
pets = [
    {"name": "Fluffy", "type": "cat", "age": 2, "favorite_toy": "ball"},
    {"name": "Rex", "type": "dog", "age": 5, "favorite_toy": "rope"},
    {"name": "Nibbles", "type": "hamster", "age": 1, "favorite_toy": "wheel"},
    {"name": "Goldie", "type": "fish", "age": 1, "favorite_toy": "castle"},
    {"name": "Whiskers", "type": "cat", "age": 3, "favorite_toy": "mouse"},
    {"name": "Buddy", "type": "dog", "age": 4, "favorite_toy": "ball"},
]

print("🐾 Pet Store Detective Investigation!")
print("\\n📋 All Our Pets:")
for pet in pets:
    print(f"🐕 {pet['name']} is a {pet['age']} year old {pet['type']} who loves {pet['favorite_toy']}")

# Group pets by type
pet_types = {}
for pet in pets:
    pet_type = pet["type"]
    if pet_type in pet_types:
        pet_types[pet_type] = pet_types[pet_type] + 1
    else:
        pet_types[pet_type] = 1

print(f"\\n📊 Pet Type Count:")
for pet_type, count in pet_types.items():
    print(f"🐾 {pet_type}: {count}")

# Find popular toys
toy_popularity = {}
for pet in pets:
    toy = pet["favorite_toy"]
    if toy in toy_popularity:
        toy_popularity[toy] = toy_popularity[toy] + 1
    else:
        toy_popularity[toy] = 1

print(f"\\n🎾 Most Popular Toys:")
for toy, count in toy_popularity.items():
    print(f"🎯 {toy}: {count} pets love it!")

# Age analysis
ages = [pet["age"] for pet in pets]
youngest_age = min(ages)
oldest_age = max(ages)
average_age = sum(ages) / len(ages)

print(f"\\n📈 Age Detective Report:")
print(f"👶 Youngest pet: {youngest_age} years old")
print(f"👴 Oldest pet: {oldest_age} years old")
print(f"📊 Average age: {average_age:.1f} years old")

# Special recommendations
print(f"\\n💡 Data Detective Recommendations:")
most_common_pet = max(pet_types, key=pet_types.get)
print(f"🏆 Most popular pet type: {most_common_pet}")
most_popular_toy = max(toy_popularity, key=toy_popularity.get)
print(f"🎾 Stock up on {most_popular_toy}s - they're the most popular!")
\`\`\`

What other questions could we answer about these pets using data detective skills?`
    },
    {
      title: "Creating Data Stories",
      content: `Great data detectives know how to tell stories with their discoveries! Let's learn how to share our findings.

\`\`\`python interactive
# School lunch data for our cafeteria investigation
lunch_choices = ["pizza", "salad", "sandwich", "soup", "pizza", "pizza", "salad", "sandwich", "pizza", "soup", "sandwich", "pizza"]

print("🍕 Cafeteria Data Detective Story!")
print("="*40)

# Chapter 1: The Mystery
print("\\n📖 Chapter 1: The Mystery")
print("The cafeteria wants to know what food to make more of.")
print("Our mission: Find out what students like to eat!")

# Chapter 2: Collecting Evidence
print("\\n📖 Chapter 2: Collecting the Evidence")
print(f"We surveyed {len(lunch_choices)} students about their lunch choices.")
print("Here's what they told us:", lunch_choices)

# Chapter 3: Analyzing the Clues
print("\\n📖 Chapter 3: Analyzing the Clues")
food_counts = {}
for food in lunch_choices:
    if food in food_counts:
        food_counts[food] = food_counts[food] + 1
    else:
        food_counts[food] = 1

for food, count in food_counts.items():
    percentage = (count / len(lunch_choices)) * 100
    print(f"🍽️ {food}: {count} students ({percentage:.1f}%)")

# Chapter 4: The Big Discovery
print("\\n📖 Chapter 4: The Big Discovery!")
winner = max(food_counts, key=food_counts.get)
winner_count = food_counts[winner]
print(f"🏆 The winner is {winner}! {winner_count} out of {len(lunch_choices)} students chose it!")

# Chapter 5: Recommendations
print("\\n📖 Chapter 5: Our Recommendations")
print(f"💡 The cafeteria should make more {winner}!")
print(f"📈 They should prepare {winner_count * 2} servings for every {len(lunch_choices)} students.")

# Create a simple visual story
print("\\n🎨 Visual Story:")
for food, count in food_counts.items():
    visual = "🟩" * count
    print(f"{food}: {visual} ({count})")

print("\\n🎉 The End! We solved the cafeteria mystery using data detective skills!")
\`\`\`

Try creating your own data story about something you're curious about!`
    },
    {
      title: "Amazing Data Detective Superpowers",
      content: `Data detectives have incredible superpowers that help make the world better! Here are some amazing things data detectives do:

**🏥 Health Heroes**
- Help doctors find the best treatments for people
- Track how many people get better with different medicines
- Figure out what foods keep people healthy

**🌱 Environment Protectors**
- Count endangered animals to help save them
- Track pollution to keep our air and water clean
- Study weather patterns to predict storms

**🎮 Entertainment Engineers**
- Make video games more fun by seeing what players like
- Help movie companies know what stories people want to see
- Create music apps that suggest songs you'll love

**🚀 Space Explorers**
- Study data from telescopes to discover new planets
- Track rocket flights to other worlds
- Look for patterns in the stars

**🏫 Education Enhancers**
- Figure out the best ways to help students learn
- Create educational games that are fun and helpful
- Find out what teaching methods work best

**🛒 Shopping Assistants**
- Help stores know what products people want to buy
- Make online shopping easier and faster
- Predict what you might need before you even know it!

These data detectives use the same skills you're learning right now! 🕵️‍♀️✨`
    },
    {
      title: "Your Data Detective Toolkit",
      content: `Congratulations! You now have a data detective toolkit full of superpowers! Here's what you've learned:

**🔍 Your Detective Skills:**
- ✅ Collecting data (information) about interesting topics
- ✅ Organizing data into groups and categories
- ✅ Counting and finding patterns in your data
- ✅ Making simple charts and graphs with symbols
- ✅ Finding the biggest, smallest, and average numbers
- ✅ Telling stories with your data discoveries
- ✅ Making predictions and recommendations

**🎯 Fun Projects to Try:**
- 📊 Survey your family about their favorite movies
- 🌡️ Track the temperature for a week and predict the weather
- 🍎 Count different types of food in your kitchen
- 📚 Keep track of how many pages you read each day
- 🎨 Organize your art supplies by color and count them
- 🕰️ Track what time you go to bed and wake up

**🚀 Keep Growing Your Powers:**
- Ask lots of questions about the world around you
- Practice counting and organizing things you find interesting
- Look for patterns in games, nature, and everyday life
- Share your discoveries with friends and family
- Keep being curious about everything!

**🌟 Remember:**
Every expert data detective started exactly where you are now! The most important thing is to stay curious, ask questions, and have fun exploring the amazing world of data.

You're now officially a Data Detective! Go out there and discover amazing things! 🕵️‍♀️🎉🏆`
    }
  ]
}; 