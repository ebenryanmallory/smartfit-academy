// TEST COMPONENT - FOR TESTING PURPOSES ONLY - WILL BE REMOVED
// This component provides hardcoded lesson content for testing the GenerateTopicLessonModal

import React from 'react';
import { Button } from './ui/button';

interface TestLessonContentProps {
  onLoadTestLesson: (content: string) => void;
}

const TestLessonContent: React.FC<TestLessonContentProps> = ({ onLoadTestLesson }) => {
  const testLessonMarkdown = `# Introduction to Photosynthesis

## What is Photosynthesis?

Photosynthesis is the process by which plants, algae, and some bacteria convert light energy (usually from the sun) into chemical energy stored in glucose molecules. This process is fundamental to life on Earth as it produces the oxygen we breathe and forms the base of most food chains.

## The Photosynthesis Equation

The overall chemical equation for photosynthesis is:

\`\`\`
6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂
\`\`\`

**In words:** Carbon dioxide + Water + Light energy → Glucose + Oxygen

## Key Components Needed

### 1. **Chloroplasts**
- Organelles found in plant cells
- Contain chlorophyll, the green pigment that captures light energy
- Located mainly in leaf cells

### 2. **Chlorophyll**
- Green pigment that absorbs light energy
- Primarily absorbs red and blue light, reflects green light
- This is why plants appear green to our eyes

### 3. **Sunlight**
- Provides the energy needed to drive the chemical reactions
- Different wavelengths of light are absorbed differently

### 4. **Carbon Dioxide (CO₂)**
- Enters the plant through small pores called stomata
- Usually found on the underside of leaves

### 5. **Water (H₂O)**
- Absorbed by the roots from the soil
- Transported up through the plant via the xylem

## The Two Main Stages

### Light-Dependent Reactions (Photo)
- Occur in the thylakoid membranes of chloroplasts
- Chlorophyll absorbs light energy
- Water molecules are split, releasing oxygen as a byproduct
- Energy is captured in molecules called ATP and NADPH

### Light-Independent Reactions (Synthesis)
- Also called the Calvin Cycle
- Occur in the stroma of chloroplasts
- Use ATP and NADPH from the light reactions
- Carbon dioxide is "fixed" into glucose molecules

## Why is Photosynthesis Important?

1. **Oxygen Production**: Nearly all oxygen in our atmosphere comes from photosynthesis
2. **Food Production**: Forms the base of most food chains on Earth
3. **Carbon Dioxide Removal**: Helps regulate atmospheric CO₂ levels
4. **Energy Storage**: Converts solar energy into chemical energy that can be used later

## Fun Facts

- A large tree can produce enough oxygen for two people per day
- Photosynthesis evolved over 3 billion years ago
- Some plants can photosynthesize even in very low light conditions
- The Amazon rainforest produces about 20% of the world's oxygen

## Practice Questions

1. What are the main reactants (inputs) of photosynthesis?
2. Where in the plant cell does photosynthesis occur?
3. Why do plants appear green?
4. What would happen to life on Earth if photosynthesis stopped?

## Next Steps

Now that you understand the basics of photosynthesis, you might want to explore:
- The detailed biochemistry of the Calvin Cycle
- How different types of plants have adapted their photosynthetic processes
- The relationship between photosynthesis and cellular respiration
- Environmental factors that affect photosynthetic rates

---

*This lesson provides a foundational understanding of photosynthesis. Continue exploring to deepen your knowledge of this fascinating biological process!*`;

  const handleLoadTest = () => {
    onLoadTestLesson(testLessonMarkdown);
  };

  return (
    <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
      <p className="text-sm text-blue-700 mb-3">
        <strong>Testing Mode:</strong> Load sample lesson plan to test the modal display
      </p>
      <Button
        onClick={handleLoadTest}
        variant="outline"
        size="sm"
        className="bg-blue-100 hover:bg-blue-200 border-blue-300 text-blue-700"
      >
        Load Test Lesson Plan
      </Button>
    </div>
  );
};

export default TestLessonContent; 