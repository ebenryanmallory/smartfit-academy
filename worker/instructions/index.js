import { educationalAssistantInstructions } from './educational-assistant.js';
import { lessonPlanGeneratorInstructions } from './lesson-plan-generator.js';
import { lessonContentGeneratorInstructions } from './lesson-content-generator.js';
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
export function getInstructions(type) {
    return instructions[type];
}
// Export individual instructions for direct import
export { educationalAssistantInstructions, lessonPlanGeneratorInstructions, lessonContentGeneratorInstructions };
