// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.

import type { LessonData } from "../types";

export const lesson: LessonData = {
  id: "c-machine-learning-fundamentals",
  title: "How Computers Learn - For Young Learners",
  description: "Discover how computers can learn to recognize patterns and make predictions, just like magic!",
  sections: [
    {
      title: "What is Machine Learning?",
      content: `Machine Learning is like teaching a computer to learn new things, just like how you learn!

Think of it this way:
- When you see many pictures of cats, you learn what cats look like
- When a computer sees many pictures of cats, it also learns what cats look like!
- Then both you and the computer can recognize cats in new pictures

It's like the computer becomes smarter by looking at lots and lots of examples! 🧠✨`
    },
    {
      title: "Three Ways Computers Learn",
      content: `Just like there are different ways you learn at school, computers have different ways to learn too!

**1. Learning with a Teacher (Supervised Learning)**
- Like learning math with flashcards that show "2 + 2 = 4"
- The computer sees examples with the right answers
- Example: Teaching a computer to recognize if a picture shows a dog or a cat

**2. Learning by Finding Patterns (Unsupervised Learning)**  
- Like sorting your toys into groups without being told how
- The computer finds hidden patterns in things
- Example: Grouping songs that sound similar together

**3. Learning by Trial and Error (Reinforcement Learning)**
- Like learning to ride a bike by practicing and getting better
- The computer tries things and learns from mistakes
- Example: Teaching a computer to play a video game and get high scores

🎮 Which type sounds most fun to you?`
    },
    {
      title: "Let's Make Simple Predictions!",
      content: `Let's pretend we're teaching a computer to predict things! We'll start with something super simple.

\`\`\`python interactive
# Let's predict how tall someone is based on their age!
# (This is just pretend data)

ages = [5, 8, 10, 12, 15]
heights = [100, 120, 135, 150, 170]  # in centimeters

print("Age -> Height")
for i in range(len(ages)):
    print(f"{ages[i]} years old -> {heights[i]} cm tall")

# Simple prediction: each year you grow about 5 cm
def predict_height(age):
    return age * 5 + 75

# Let's test our prediction!
new_age = 9
predicted_height = predict_height(new_age)
print(f"\\nPrediction: A {new_age} year old is about {predicted_height} cm tall!")
\`\`\`

Try changing the age and see what height it predicts!`
    },
    {
      title: "Teaching Computers to Count",
      content: `Let's see how we can help a computer count things in lists!

\`\`\`python interactive
# Let's count our favorite things!
favorite_fruits = ["apple", "banana", "apple", "orange", "banana", "apple"]
favorite_colors = ["red", "blue", "red", "green", "blue", "red", "red"]

# Count apples
apple_count = 0
for fruit in favorite_fruits:
    if fruit == "apple":
        apple_count = apple_count + 1

print(f"We have {apple_count} apples!")

# Count red things
red_count = 0
for color in favorite_colors:
    if color == "red":
        red_count = red_count + 1

print(f"We have {red_count} red things!")

# Python has a cool way to count too!
all_fruits = favorite_fruits.count("banana")
print(f"We have {all_fruits} bananas!")
\`\`\`

Try counting different fruits or colors by changing the words!`
    },
    {
      title: "Sorting and Grouping Like a Computer",
      content: `Computers are really good at sorting and grouping things. Let's try it!

\`\`\`python interactive
# Let's sort our pets by type!
pets = ["dog", "cat", "dog", "bird", "cat", "fish", "dog"]

# Count each type of pet
pet_groups = {}

for pet in pets:
    if pet in pet_groups:
        pet_groups[pet] = pet_groups[pet] + 1
    else:
        pet_groups[pet] = 1

print("Our pet groups:")
for pet_type, count in pet_groups.items():
    print(f"🐾 {pet_type}: {count}")

# Let's sort numbers from smallest to biggest
numbers = [7, 2, 9, 1, 5, 8, 3]
print(f"\\nBefore sorting: {numbers}")

sorted_numbers = sorted(numbers)
print(f"After sorting: {sorted_numbers}")

# Find the biggest and smallest
biggest = max(numbers)
smallest = min(numbers)
print(f"\\nBiggest number: {biggest}")
print(f"Smallest number: {smallest}")
\`\`\`

Try adding your own pets or numbers to see how the computer sorts them!`
    },
    {
      title: "Making a Smart Guessing Game",
      content: `Let's create a game where the computer learns to make better guesses!

\`\`\`python interactive
def smart_guessing_game():
    # The computer remembers what worked before
    good_guesses = [5, 7, 8]  # Numbers that were close before
    
    print("🎯 Smart Guessing Game!")
    print("Think of a number between 1 and 10...")
    
    secret_number = 6  # Pretend this is your secret number
    
    # Computer makes a smart guess based on what worked before
    computer_guess = good_guesses[0]  # Start with first good guess
    
    print(f"🤖 Computer guesses: {computer_guess}")
    
    if computer_guess == secret_number:
        print("🎉 WOW! The computer got it right!")
        print("The computer is learning to make better guesses!")
    elif computer_guess < secret_number:
        print("📈 Too low! The computer will remember to guess higher next time.")
        # Computer learns: add higher numbers to good_guesses
        good_guesses.append(computer_guess + 2)
    else:
        print("📉 Too high! The computer will remember to guess lower next time.")
        # Computer learns: add lower numbers to good_guesses
        good_guesses.append(computer_guess - 2)
    
    print(f"Computer's memory of good guesses: {good_guesses}")

# Play the game!
smart_guessing_game()

print("\\n🌟 This is how machine learning works!")
print("The computer remembers what worked and gets better over time!")
\`\`\`

Try changing the secret number to see how the computer learns!`
    },
    {
      title: "Amazing Things Computers Can Learn",
      content: `Machine learning helps computers do amazing things that make our world better!

**🏥 Helping Doctors**
- Computers can look at X-rays and help doctors find problems
- They can help discover new medicines faster

**🎵 Music and Entertainment**
- Apps like Spotify learn what music you like and suggest new songs
- Video games become more fun with AI characters

**🚗 Transportation**
- Some cars can drive themselves by learning about roads
- Apps help find the fastest way to get places

**🌍 Protecting Our Planet**
- Computers help scientists study climate change
- They help track and protect endangered animals

**📚 Learning and Education**
- Educational apps learn how you like to study
- They can help make learning more fun and personalized

**🎨 Art and Creativity**
- AI can help create beautiful art and music
- It can help write stories and poems

Machine learning is like giving computers superpowers to help make the world a better place! And guess what? You can learn to create these superpowers too! 🦸‍♀️🦸‍♂️`
    },
    {
      title: "You Can Be a Machine Learning Hero!",
      content: `Want to become a machine learning hero? Here's how you can start your adventure!

**🎮 Fun Ways to Learn:**
- Play coding games like Scratch or Code.org
- Try apps that teach programming with puzzles
- Watch videos about how computers learn

**🧮 Math is Your Superpower:**
- Practice counting, adding, and finding patterns
- Learn about graphs and charts
- Math helps you understand how computers think!

**🔍 Be Curious:**
- Ask "How does this work?" about apps and games
- Try to spot patterns in everyday things
- Experiment with simple coding projects

**📖 Cool Projects to Try:**
- Create a simple guessing game (like we did!)
- Make a program that sorts your favorite things
- Build a calculator that does math for you

**🤝 Share and Learn:**
- Join coding clubs at school
- Share your projects with friends and family
- Help others learn what you've discovered

Remember: Every expert was once a beginner! 🌱

The most important thing is to have fun while learning. Machine learning is like having a magic toolbox that can solve problems and create amazing things. And YOU can learn to use this magic! ✨🎁`
    }
  ]
}; 