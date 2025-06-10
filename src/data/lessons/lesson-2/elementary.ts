// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.

import type { LessonData } from "../types";

export const lesson2: LessonData = {
  id: 2,
  title: "Let's Learn to Code! - For Young Learners",
  description: "Discover how to give instructions to computers using Python programming!",
  sections: [
    {
      title: "What is Programming?",
      content: `Programming is like giving instructions to a computer! Just like you follow a recipe to make cookies, computers follow our instructions called "code" to do things.

Python is a programming language that's perfect for beginners. It's named after a funny snake! 🐍`
    },
    {
      title: "Variables - Computer's Memory Boxes",
      content: `Variables are like boxes where we store information for the computer to remember!

Think of it like this:
- A box labeled "name" might hold "Emma"
- A box labeled "age" might hold the number 8
- A box labeled "favorite_color" might hold "blue"

Let's try it:

\`\`\`python interactive
name = "Alex"
age = 9
favorite_color = "red"
favorite_number = 7

print("Hi, my name is", name)
print("I am", age, "years old")
print("My favorite color is", favorite_color)
print("My lucky number is", favorite_number)
\`\`\`

Try changing the values to your own information!`
    },
    {
      title: "Math with Computers",
      content: `Computers are super good at math! They can add, subtract, multiply, and divide really fast.

Let's do some math with Python:

\`\`\`python interactive
# Addition
toys = 5 + 3
print("I have", toys, "toys")

# Subtraction  
cookies_left = 10 - 4
print("I have", cookies_left, "cookies left")

# Multiplication
stickers = 6 * 4
print("I have", stickers, "stickers total")

# Let's try with your numbers!
my_number = 7
doubled = my_number * 2
print(my_number, "times 2 equals", doubled)
\`\`\`

Try changing the numbers and see what happens!`
    },
    {
      title: "Making Decisions with If/Else",
      content: `Sometimes we want the computer to make choices, just like we do!

\`\`\`python interactive
age = 8

if age >= 10:
    print("You can ride the big roller coaster!")
else:
    print("You need to wait a bit more to ride the big one!")

# Let's try with different ages
age = 12
if age >= 10:
    print("Hooray! You're old enough for the big roller coaster!")
else:
    print("Don't worry, the smaller rides are fun too!")
\`\`\`

Try changing the age to see different messages!`
    },
    {
      title: "Counting with Loops",
      content: `Loops help us do the same thing many times without writing the same code over and over!

\`\`\`python interactive
# Let's count to 5!
print("Counting to 5:")
for number in range(1, 6):
    print(number)

print("Blast off! 🚀")

# Let's say hello to our friends
friends = ["Emma", "Sam", "Maya", "Alex"]
for friend in friends:
    print("Hello", friend + "!")
\`\`\`

Try adding your friends' names to the list!`
    },
    {
      title: "Creating Our Own Instructions (Functions)",
      content: `Functions are like creating our own special instructions that we can use again and again!

\`\`\`python interactive
def say_happy_birthday(name):
    print("🎉 Happy Birthday", name + "! 🎂")
    print("Hope your day is super awesome!")

# Let's use our function
say_happy_birthday("Sarah")
say_happy_birthday("Mike")

# Make a function to calculate how many legs animals have
def count_legs(animal, leg_count):
    print("A", animal, "has", leg_count, "legs!")

count_legs("dog", 4)
count_legs("spider", 8)
count_legs("bird", 2)
\`\`\`

Try creating a function to say hello in a fun way!`
    },
    {
      title: "Let's Build Something Fun!",
      content: `Now let's put everything together and make a simple guessing game!

\`\`\`python interactive
def number_guessing_game():
    secret_number = 7
    print("I'm thinking of a number between 1 and 10!")
    
    guess = 5  # You can change this number
    
    if guess == secret_number:
        print("WOW! You guessed it! The number was", secret_number)
    elif guess < secret_number:
        print("Your guess is too low! Try a bigger number.")
    else:
        print("Your guess is too high! Try a smaller number.")

# Play the game!
number_guessing_game()
\`\`\`

Try changing the guess number to see different responses!

🎯 **Challenge**: Can you make the game with a different secret number?`
    }
  ]
}; 