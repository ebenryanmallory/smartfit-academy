import { educationalAssistantInstructions } from './educational-assistant';

// Export all instruction sets
export const instructions = {
  educationalAssistant: educationalAssistantInstructions,
  // Future instruction sets can be added here
  // lessonTutor: lessonTutorInstructions,
  // codeReviewer: codeReviewerInstructions,
  // etc.
};

// Helper function to get instruction by type
export function getInstructions(type: keyof typeof instructions): string {
  return instructions[type];
}

// Export individual instructions for direct import
export { educationalAssistantInstructions }; 