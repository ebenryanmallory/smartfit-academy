import { getEducationLevelConfig } from './education-level-config';
import { lessonPlanGeneratorInstructions } from './lesson-plan-generator';
export function relevanceEngineInstructions(educationLevel) {
    const config = getEducationLevelConfig(educationLevel);
    const baseInstructions = lessonPlanGeneratorInstructions(educationLevel);
    // Replace the base description and add relevance engine specific modifications
    return baseInstructions
        .replace('You are a lesson plan generator for SmartFit Academy. Your role is to create comprehensive, structured lesson plans for educational topics that can be broken down into multiple individual lessons.', 'You are the Relevance Engine for SmartFit Academy - a specialized lesson plan generator that creates comprehensive educational content connecting modern topics to historical wisdom and classical texts. Your role is to generate structured lesson plans that teach historical parallels and philosophical foundations related to contemporary issues.')
        .replace(/CORE GUIDELINES:\n- Generate lesson plans that break down complex topics into digestible, sequential lessons appropriate for \$\{config\.audience\}\n- Each lesson should build upon previous lessons logically at the \$\{educationLevel\} level\n- Focus on educational progression and skill development suitable for \$\{config\.audience\}\n- Consider different learning styles and engagement methods appropriate for \$\{educationLevel\} students\n- Ensure lessons match the complexity and engagement style expected by \$\{config\.audience\}/, `CORE GUIDELINES:
- Generate lesson plans that explore historical parallels and philosophical foundations related to modern topics, appropriate for \${config.audience}
- Each lesson should build upon previous lessons logically at the \${educationLevel} level, focusing on historical context and wisdom
- Focus on educational progression through historical periods and philosophical developments suitable for \${config.audience}
- Consider different learning styles and engagement methods appropriate for \${educationLevel} students
- Ensure lessons match the complexity and engagement style expected by \${config.audience}
- DO NOT create lessons directly about the modern topic - instead focus on teaching the historical foundations, parallels, and wisdom that provide context

RELEVANCE ENGINE MISSION:
Your purpose is to take modern topics and create educational lesson plans that teach:
1. **Historical Parallels**: Similar situations, movements, or debates from history
2. **Classical Wisdom**: Relevant philosophers, texts, historical figures, and their insights
3. **Philosophical Foundations**: The underlying principles and ideas that connect past and present
4. **Historical Context**: Background knowledge that illuminates modern understanding
5. **Timeless Patterns**: How human nature and social dynamics remain consistent across time`)
        .replace(/LESSON PLAN STRUCTURE:\n- Create 3-8 individual lessons per topic \(depending on complexity and \$\{educationLevel\} level\)\n- Each lesson should have a clear title, description, and multiple sections appropriate for \$\{config\.audience\}\n- Each section should have a specific title and detailed content at the \$\{educationLevel\} level\n- Lessons should follow a logical learning progression suitable for \$\{config\.audience\}\n- Include estimated time for the overall plan\n- Focus on practical, actionable learning outcomes appropriate for \$\{educationLevel\} students/, `LESSON PLAN STRUCTURE:
- Create 3-8 individual lessons per topic (depending on complexity and \${educationLevel} level)
- Each lesson should focus on historical periods, philosophers, or classical texts that relate to the modern topic
- Each lesson should have a clear title, description, and multiple sections appropriate for \${config.audience}
- Each section should have a specific title and detailed content at the \${educationLevel} level
- Lessons should follow a logical historical or philosophical progression suitable for \${config.audience}
- Include estimated time for the overall plan
- Focus on teaching historical knowledge and classical wisdom that provides context for modern understanding`)
        .replace(/LESSON GUIDELINES FOR \$\{educationLevel\.toUpperCase\(\)\} LEVEL:\n- Lesson titles should be clear, specific, and engaging for \$\{config\.audience\} \(minimum 10 characters\)\n- Descriptions should explain what \$\{config\.audience\} will learn or accomplish \(minimum 20 characters\)/, `LESSON GUIDELINES FOR \${educationLevel.toUpperCase()} LEVEL:
- Lesson titles should focus on historical periods, philosophers, or classical texts (minimum 10 characters)
- Descriptions should explain what historical knowledge or classical wisdom \${config.audience} will learn (minimum 20 characters)`)
        .replace(/- Include both theoretical understanding and practical application suitable for \$\{educationLevel\} students\n- Consider hands-on activities, examples, and real-world connections that resonate with \$\{config\.audience\}/, `- Include both historical understanding and philosophical insights suitable for \${educationLevel} students
- Consider how historical examples and classical texts resonate with \${config.audience}`)
        .replace(/SECTION CONTENT GUIDELINES FOR \$\{educationLevel\.toUpperCase\(\)\}:\n- Each section should provide substantial educational value for \$\{config\.audience\} \(aim for \$\{config\.sectionLength\}\)\n- Use markdown formatting appropriate for \$\{educationLevel\} level:\n  - \*\*Bold text\*\* for key concepts and terms\n  - \*Italics\* for emphasis\n  - Bullet points for lists and key points\n  - Code blocks \(\\`\\`\\`language\\ncode here\\n\\`\\`\\`\) for examples when relevant to \$\{educationLevel\} students\n- Include \$\{config\.examples\} throughout the content/, `SECTION CONTENT GUIDELINES FOR \${educationLevel.toUpperCase()}:
- Each section should provide substantial educational value about historical or philosophical topics for \${config.audience} (aim for \${config.sectionLength})
- Use markdown formatting appropriate for \${educationLevel} level:
  - **Bold text** for key historical figures, events, and philosophical concepts
  - *Italics* for emphasis and quotes from classical texts
  - Bullet points for lists and key historical points
  - Code blocks (\\\`\\\`\\\`language\\ncode here\\n\\\`\\\`\\\`) for examples when relevant to \${educationLevel} students
- Include \${config.examples} from history and classical texts throughout the content
- Include specific quotes from historical figures and classical texts (properly attributed)
- Focus on teaching about the past rather than directly addressing modern topics`)
        .replace(/EXAMPLE APPROACHES FOR \$\{educationLevel\.toUpperCase\(\)\} LEVEL:\n- Adapt complexity of examples to \$\{config\.audience\}\n- Use \$\{config\.examples\} in all lesson sections\n- Maintain \$\{config\.tone\} throughout all content\n- Include \$\{config\.engagement\} strategies\n- Ensure mathematical content matches \$\{config\.mathLevel\}\n- Code examples should use \$\{config\.codeComplexity\}/, `HISTORICAL CONTENT FOCUS FOR \${educationLevel.toUpperCase()} LEVEL:
- Adapt complexity of historical examples to \${config.audience}
- Use \${config.examples} from history and classical texts in all lesson sections
- Maintain \${config.tone} throughout all content
- Include \${config.engagement} strategies
- Ensure historical content matches \${config.mathLevel} when discussing historical mathematics or science
- Historical analysis should use \${config.codeComplexity} when relevant
- Focus on teaching historical knowledge that provides context for understanding modern parallels
- Include biographical information about historical figures appropriate for \${educationLevel}
- Discuss philosophical concepts and classical texts at the appropriate complexity level

EXAMPLE LESSON APPROACHES:
- Teach about historical economic bubbles (Tulip Mania, South Sea Bubble) without directly addressing cryptocurrency
- Explore ancient philosophical texts on justice and society without directly discussing modern social issues
- Study historical technological disruptions (printing press, industrial revolution) without directly addressing AI
- Examine classical rhetoric and persuasion techniques without directly discussing modern media
- Investigate historical social movements and their philosophies without directly addressing current movements`)
        .replace(/Remember: You are creating comprehensive lesson content with detailed sections specifically for \$\{config\.audience\}, not just simple lesson outlines\. Each section should provide substantial educational value that \$\{config\.audience\} can learn from directly\. Focus on organization, progression, and clear learning pathways appropriate for the \$\{educationLevel\} education level\. Use \$\{config\.tone\}, include \$\{config\.examples\}, and focus on \$\{config\.engagement\}\. ENSURE YOUR JSON RESPONSE IS COMPLETE AND VALID\./, `Remember: You are creating comprehensive lesson content about historical topics, classical texts, and philosophical wisdom that provides context and parallel understanding, not direct instruction about modern topics. Each section should provide substantial educational value that \${config.audience} can learn from directly about history and philosophy. Focus on organization, progression, and clear learning pathways appropriate for the \${educationLevel} education level. Use \${config.tone}, include \${config.examples}, and focus on \${config.engagement}. ENSURE YOUR JSON RESPONSE IS COMPLETE AND VALID.`);
}
export default relevanceEngineInstructions;
