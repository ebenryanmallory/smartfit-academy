import { educationalAssistantInstructions } from './educational-assistant';
import { lessonPlanGeneratorInstructions } from './lesson-plan-generator';
import { lessonContentGeneratorInstructions } from './lesson-content-generator';
import { relevanceEngineInstructions } from './relevance-engine-generator';
import { historicalConnectionGeneratorInstructions } from './historical-connection-generator';

// Export all instruction sets
export const instructions = {
  educationalAssistant: educationalAssistantInstructions,
  // Note: lessonPlanGenerator and lessonContentGenerator are now functions
  // They should be called directly with an education level parameter
  // Future instruction sets can be added here
  // lessonTutor: lessonTutorInstructions,
  // codeReviewer: codeReviewerInstructions,
  // etc.
};

// Helper function to get instruction by type (only for constant instructions)
export function getInstructions(type: 'educationalAssistant'): string {
  return instructions[type];
}

// Export individual instructions for direct import
export { 
  educationalAssistantInstructions,
  lessonPlanGeneratorInstructions,
  lessonContentGeneratorInstructions,
  relevanceEngineInstructions,
  historicalConnectionGeneratorInstructions
}; 