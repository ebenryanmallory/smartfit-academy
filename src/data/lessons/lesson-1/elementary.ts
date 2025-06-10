// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.

import type { LessonData } from "../types";

export const lesson1: LessonData = {
  id: 1,
  title: "What is AI? - For Young Learners",
  description: "Discover what Artificial Intelligence is and how it helps us every day!",
  sections: [
    {
      title: "What is AI?",
      content: `AI stands for Artificial Intelligence. It's like making computers really smart so they can help us with things!

Think of AI like a super smart robot friend that can:
- Help you find your favorite videos
- Translate words from other languages
- Play games with you
- Help doctors take care of people`
    },
    {
      title: "AI is Everywhere!",
      content: `You already use AI every day! Here are some examples:

🎵 **Music Apps**: When you listen to music, AI helps pick songs you might like
📱 **Voice Assistants**: When you say "Hey Siri" or "OK Google", that's AI listening to you
🎮 **Video Games**: The characters you play against are controlled by AI
📺 **Video Recommendations**: AI helps find new videos you might want to watch`
    },
    {
      title: "How Does AI Learn?",
      content: `AI learns kind of like how you learn, but much faster! 

Just like you learn to recognize your friends' faces by seeing them many times, AI learns by looking at lots and lots of examples.

For example:
- To recognize cats, AI looks at thousands of cat pictures
- To understand speech, AI listens to many people talking
- To play games, AI practices millions of times`
    },
    {
      title: "Let's Try Simple Coding!",
      content: `Here's a fun way to think like a computer. We can write simple instructions:

\`\`\`python interactive
# This is like giving instructions to a computer
def say_hello(name):
    return f"Hello, {name}! Welcome to AI learning!"

# Try it with your name!
print(say_hello("Alex"))
print(say_hello("Sam"))

# You can change the names to yours and your friends' names!
\`\`\`

Try changing "Alex" and "Sam" to your name and your friend's name!`
    },
    {
      title: "AI Helps Make Our World Better",
      content: `AI is like having a super helper that never gets tired! It helps:

🏥 **Doctors**: Find diseases faster to help sick people
🌍 **Scientists**: Study the Earth and space
🚗 **Cars**: Drive more safely
📚 **Teachers**: Make learning more fun and personalized`
    },
    {
      title: "You Can Learn AI Too!",
      content: `Learning about AI is like learning a new superpower! You can start by:

- Playing with coding games and apps
- Learning basic math (AI uses lots of math!)
- Being curious and asking questions
- Practicing problem-solving

Remember: Every expert was once a beginner. You can become an AI expert too if you keep learning and practicing!`
    }
  ]
}; 