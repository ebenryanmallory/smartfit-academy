export type EducationLevel = 'elementary' | 'highschool' | 'undergrad' | 'grad';

export interface EducationLevelConfig {
  audience: string;
  tone: string;
  vocabulary: string;
  examples: string;
  complexity: string;
  codeComplexity: string;
  mathLevel: string;
  sectionLength: string;
  engagement: string;
}

export const educationLevelConfigs: Record<EducationLevel, EducationLevelConfig> = {
  elementary: {
    audience: 'elementary school students (ages 6-11)',
    tone: 'simple, fun, and encouraging',
    vocabulary: 'basic vocabulary with simple explanations',
    examples: 'relatable everyday examples and analogies',
    complexity: 'very basic concepts with lots of visual descriptions',
    codeComplexity: 'simple, interactive examples with clear comments',
    mathLevel: 'basic arithmetic and simple patterns',
    sectionLength: '100-300 characters per section',
    engagement: 'emojis, fun facts, and interactive elements'
  },
  highschool: {
    audience: 'high school students (ages 14-18)',
    tone: 'engaging and informative',
    vocabulary: 'age-appropriate with technical terms explained clearly',
    examples: 'real-world applications and career connections',
    complexity: 'intermediate concepts with practical applications',
    codeComplexity: 'moderate programming examples with explanations',
    mathLevel: 'algebra, basic statistics, and logical reasoning',
    sectionLength: '300-800 characters per section',
    engagement: 'career insights, hands-on projects, and future planning'
  },
  undergrad: {
    audience: 'undergraduate students (ages 18-22)',
    tone: 'academic yet accessible',
    vocabulary: 'technical terminology with proper definitions',
    examples: 'industry applications and theoretical foundations',
    complexity: 'comprehensive coverage with depth and breadth',
    codeComplexity: 'substantial programming examples with best practices',
    mathLevel: 'calculus, linear algebra, statistics, and probability',
    sectionLength: '500-2000 characters per section',
    engagement: 'practical implementations, case studies, and research connections'
  },
  grad: {
    audience: 'graduate students and researchers (ages 22+)',
    tone: 'scholarly and rigorous',
    vocabulary: 'advanced technical terminology and academic language',
    examples: 'cutting-edge research and theoretical frameworks',
    complexity: 'advanced theoretical concepts with mathematical rigor',
    codeComplexity: 'sophisticated implementations with theoretical foundations',
    mathLevel: 'advanced mathematics, optimization theory, and formal methods',
    sectionLength: '800-3000 characters per section',
    engagement: 'research frontiers, philosophical implications, and open problems'
  }
};

export function getEducationLevelConfig(educationLevel: EducationLevel): EducationLevelConfig {
  return educationLevelConfigs[educationLevel];
}

export function isValidEducationLevel(level: string): level is EducationLevel {
  return ['elementary', 'highschool', 'undergrad', 'grad'].includes(level);
} 