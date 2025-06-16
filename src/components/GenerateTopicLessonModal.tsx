import React, { useState, useEffect, useCallback } from 'react';
import { useUser, useAuth } from '@clerk/clerk-react';
import { toast } from 'sonner';
import { generateUserLessonId } from '@/utils/lessonIdUtils';
import { getTestPrepLessonPlans } from '@/data/test-prep/lessonPlans';

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from './ui/dialog';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Loader2, BookOpen, X, ChevronDown, ChevronRight, Trash2, RefreshCw, Plus, Minus, GraduationCap } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

type EducationLevel = 'elementary' | 'highschool' | 'undergrad' | 'grad';

// Function to get display name for education level
const getEducationLevelDisplayName = (level: EducationLevel): string => {
  switch (level) {
    case 'elementary':
      return 'Elementary School';
    case 'highschool':
      return 'High School';
    case 'undergrad':
      return 'Undergraduate';
    case 'grad':
      return 'Graduate';
    default:
      return 'Undergraduate';
  }
};

interface LessonSection {
  title: string;
  content: string;
}

interface Lesson {
  id: string;
  title: string;
  description: string;
  sections?: LessonSection[]; // Generated lesson sections
  content?: string; // Loaded on demand for individual lesson content
  isExpanded: boolean;
  isLoadingContent: boolean;
}

interface LessonPlan {
  lessons: Lesson[];
  totalEstimatedTime: string;
}

interface GenerateTopicLessonModalProps {
  isOpen: boolean;
  onClose: () => void;
  topic: string;
}



// Function to save multiple lesson plans directly to user data
const saveTestPrepLessonPlans = async (topic: string, user: any, getToken: any) => {
  const lessonPlansData = getTestPrepLessonPlans(topic);
  
  if (lessonPlansData.length === 0) {
    throw new Error('No lesson plans found for this topic');
  }

  const token = await getToken();
  if (!token) {
    throw new Error('Failed to get authentication token');
  }

  // Save each lesson plan separately
  const savePromises = lessonPlansData.map(async (planData, index) => {
    const lessonPlanData = {
      topic: planData.title, // Use the specific lesson plan title as topic
      title: `${planData.title} - Study Plan`,
      totalEstimatedTime: `${planData.lessons.length * 2}-${planData.lessons.length * 3} hours`,
      uuid: generateUserLessonId(),
      lessons: planData.lessons.map((lessonTitle, lessonIndex) => ({
        title: lessonTitle,
        description: `Comprehensive lesson covering ${lessonTitle.toLowerCase()}`,
        content: null, // Will be generated when expanded
        uuid: generateUserLessonId(),
        lesson_order: lessonIndex + 1
      }))
    };

    const response = await fetch('/api/d1/user/lesson-plans', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: JSON.stringify(lessonPlanData),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Failed to save lesson plan ${index + 1}:`, response.status, errorText);
      throw new Error(`Failed to save lesson plan: ${planData.title}`);
    }

    return response.json();
  });

  // Wait for all lesson plans to be saved
  await Promise.all(savePromises);
  
  return lessonPlansData.length;
};

const GenerateTopicLessonModal: React.FC<GenerateTopicLessonModalProps> = ({
  isOpen,
  onClose,
  topic,
}) => {
  const { user } = useUser();
  const { getToken } = useAuth();
  const [lessonPlan, setLessonPlan] = useState<LessonPlan | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [saving, setSaving] = useState(false);
  const [userEducationLevel, setUserEducationLevel] = useState<EducationLevel>('undergrad'); // Default to undergrad

  // Get pre-populated lesson plan data for test prep topics
  const testPrepLessonPlans = getTestPrepLessonPlans(topic);
  const isTestPrep = testPrepLessonPlans.length > 0;
  const [showTestPrepPreview, setShowTestPrepPreview] = useState(false);

  // Function to create and save test prep lesson plans
  const createTestPrepPlans = async () => {
    if (!user) {
      toast.error('Please sign in to create lesson plans');
      return;
    }

    setShowTestPrepPreview(false);
    setSaving(true);

    try {
      const savedCount = await saveTestPrepLessonPlans(topic, user, getToken);
      
      toast.success(`Successfully created ${savedCount} lesson plans!`, {
        description: 'Your comprehensive study plans are now available in your dashboard.',
      });
      
      // Close the modal after successful save
      onClose();
    } catch (err) {
      console.error('Error creating test prep lesson plans:', err);
      toast.error('Failed to create lesson plans. Please try again.');
      setShowTestPrepPreview(true); // Show preview again on error
    } finally {
      setSaving(false);
    }
  };

  // Fetch user's education level
  const fetchUserEducationLevel = useCallback(async () => {
    if (!user) return;

    try {
      const token = await getToken();
      if (!token) return;

      const response = await fetch('/api/d1/user', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });

      if (response.ok) {
        const data = await response.json();
        if (data.user?.education_level) {
          setUserEducationLevel(data.user.education_level as EducationLevel);
          console.log('User education level:', data.user.education_level);
        }
      }
    } catch (error) {
      console.error('Error fetching user education level:', error);
      // Keep default value on error
    }
  }, [user, getToken]);

  // Fetch education level when user changes or modal opens
  useEffect(() => {
    if (isOpen && user) {
      fetchUserEducationLevel();
    }
  }, [isOpen, user, fetchUserEducationLevel]);

  const generateLessonPlanWithStreaming = useCallback(async (retryCount = 0) => {
    if (!user) {
      toast.error('Please sign in to generate lesson plans');
      return;
    }

    setLoading(true);
    setError(null);
    setLessonPlan(null);

    try {
      const token = await getToken();
      if (!token) {
        throw new Error('Failed to get authentication token');
      }

      const response = await fetch('/llm/llama3', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          messages: [
            {
              role: 'user',
              content: `Create a comprehensive lesson plan for the topic: "${topic}". 
              
              Please generate a structured lesson plan that breaks this topic down into multiple individual lessons.
              
              Tailor the content complexity for ${getEducationLevelDisplayName(userEducationLevel)} students.
              
              IMPORTANT: Respond with ONLY valid, complete JSON. Do not truncate your response.`
            }
          ],
          instructionType: 'lessonPlanGenerator',
          educationLevel: userEducationLevel
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to generate lesson plan: ${response.status}`);
      }

      // Handle the response - check if it's streaming or regular JSON
      const responseData = await response.json();
      
      // Extract the actual lesson plan data from the response
      let planDataString = '';
      if (responseData.result && responseData.result.response) {
        // Server returned {result: {response: "..."}} format
        planDataString = responseData.result.response;
      } else if (responseData.response) {
        // Server returned {response: "..."} format
        planDataString = responseData.response;
      } else {
        // Direct response format
        planDataString = JSON.stringify(responseData);
      }

      // Parse the lesson plan JSON
      const planData = JSON.parse(planDataString);
      
      // Comprehensive validation of the parsed data
      if (!planData || typeof planData !== 'object') {
        throw new Error('Response is not a valid object');
      }
      
      if (!planData.lessonPlan || typeof planData.lessonPlan !== 'object') {
        throw new Error('Missing or invalid lessonPlan object in response');
      }
      
      if (!planData.lessonPlan.lessons || !Array.isArray(planData.lessonPlan.lessons)) {
        throw new Error('Missing or invalid lessons array in response');
      }
      
      if (planData.lessonPlan.lessons.length === 0) {
        throw new Error('No lessons found in the response');
      }
      
      // Validate each lesson
      const validatedLessons = [];
      for (let i = 0; i < planData.lessonPlan.lessons.length; i++) {
        const lesson = planData.lessonPlan.lessons[i];
        
        if (!lesson || typeof lesson !== 'object') {
          console.warn(`Skipping invalid lesson at index ${i}:`, lesson);
          continue;
        }
        
        if (!lesson.title || typeof lesson.title !== 'string' || lesson.title.trim().length === 0) {
          console.warn(`Skipping lesson ${i + 1} with invalid title:`, lesson);
          continue;
        }
        
        if (!lesson.description || typeof lesson.description !== 'string' || lesson.description.trim().length === 0) {
          console.warn(`Skipping lesson ${i + 1} with invalid description:`, lesson);
          continue;
        }
        
        // Check for obviously truncated content
        if (lesson.title.length < 5) {
          console.warn(`Lesson ${i + 1} title appears truncated:`, lesson.title);
          continue;
        }
        
        if (lesson.description.length < 10) {
          console.warn(`Lesson ${i + 1} description appears truncated:`, lesson.description);
          continue;
        }
        
        // Validate sections array
        if (!lesson.sections || !Array.isArray(lesson.sections)) {
          console.warn(`Skipping lesson ${i + 1} with invalid sections:`, lesson);
          continue;
        }
        
        if (lesson.sections.length === 0) {
          console.warn(`Skipping lesson ${i + 1} with no sections:`, lesson);
          continue;
        }
        
        // Validate and process sections
        const validatedSections = [];
        for (let j = 0; j < lesson.sections.length; j++) {
          const section = lesson.sections[j];
          
          if (!section || typeof section !== 'object') {
            console.warn(`Skipping invalid section ${j + 1} in lesson ${i + 1}:`, section);
            continue;
          }
          
          if (!section.title || typeof section.title !== 'string' || section.title.trim().length === 0) {
            console.warn(`Skipping section ${j + 1} in lesson ${i + 1} with invalid title:`, section);
            continue;
          }
          
          if (!section.content || typeof section.content !== 'string' || section.content.trim().length === 0) {
            console.warn(`Skipping section ${j + 1} in lesson ${i + 1} with invalid content:`, section);
            continue;
          }
          
          // Check for truncated sections
          if (section.title.length < 3) {
            console.warn(`Section ${j + 1} in lesson ${i + 1} title appears truncated:`, section.title);
            continue;
          }
          
          if (section.content.length < 20) {
            console.warn(`Section ${j + 1} in lesson ${i + 1} content appears truncated:`, section.content);
            continue;
          }
          
          validatedSections.push({
            title: section.title.trim(),
            content: section.content.trim()
          });
        }
        
        // Only include lesson if it has at least one valid section
        if (validatedSections.length === 0) {
          console.warn(`Skipping lesson ${i + 1} as it has no valid sections`);
          continue;
        }
        
        validatedLessons.push({
          id: generateUserLessonId(),
          title: lesson.title.trim(),
          description: lesson.description.trim(),
          sections: validatedSections,
          content: undefined, // Will be generated from sections when needed
          isExpanded: false,
          isLoadingContent: false,
        });
      }
      
      if (validatedLessons.length === 0) {
        throw new Error('No valid lessons found after validation');
      }

      const validatedPlan = {
        lessons: validatedLessons,
        totalEstimatedTime: planData.lessonPlan.totalEstimatedTime || 'Not specified',
      };

      setLessonPlan(validatedPlan);
      toast.success(`Lesson plan generated successfully with ${validatedLessons.length} lessons!`);
      
    } catch (err) {
      console.error('Error generating lesson plan:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(`Failed to generate lesson plan: ${errorMessage}`);
      toast.error('Failed to generate lesson plan', {
        description: errorMessage.length > 100 ? errorMessage.substring(0, 100) + '...' : errorMessage
      });
    } finally {
      setLoading(false);
    }
  }, [user, getToken, topic, userEducationLevel]);

  // Wrapper function for event handlers
  const generateLessonPlan = useCallback(() => {
    return generateLessonPlanWithStreaming(0);
  }, [generateLessonPlanWithStreaming]);

  const loadLessonContent = useCallback(async (lessonId: string) => {
    if (!user || !lessonPlan) return;

    const lessonIndex = lessonPlan.lessons.findIndex(l => l.id === lessonId);
    if (lessonIndex === -1) return;

    // Update lesson to show loading state
    setLessonPlan(prev => {
      if (!prev) return prev;
      const updated = { ...prev };
      updated.lessons[lessonIndex] = { ...updated.lessons[lessonIndex], isLoadingContent: true };
      return updated;
    });

    try {
      // Make API call to load lesson content
      const token = await getToken();
      if (!token) throw new Error('Failed to get authentication token');

      const response = await fetch('/llm/llama3', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          messages: [
            {
              role: 'user',
              content: `Create detailed lesson content for:

**Topic:** ${topic}
**Lesson Title:** ${lessonPlan.lessons[lessonIndex].title}
**Lesson Description:** ${lessonPlan.lessons[lessonIndex].description || 'No specific description provided'}

${lessonId.startsWith('test-prep-') ? `
**IMPORTANT: This is a standardized test preparation lesson.**
${topic.toLowerCase().includes('ged') ? 'Focus on GED test strategies, question types, and practice problems that mirror the actual GED exam format.' : ''}
${topic.toLowerCase().includes('sat') ? 'Focus on SAT test strategies, question formats, timing tips, and practice problems that mirror the actual SAT exam.' : ''}
${topic.toLowerCase().includes('act') ? 'Focus on ACT test strategies, question types, pacing techniques, and practice problems that mirror the actual ACT exam.' : ''}

Include:
- Specific test-taking strategies
- Sample questions in the actual test format
- Time management tips
- Common mistakes to avoid
- Practice exercises with detailed explanations
` : ''}

Please generate comprehensive, educational content in markdown format for this specific lesson. The content should be engaging, informative, and appropriate for ${getEducationLevelDisplayName(userEducationLevel)} students.

Include practical examples, clear explanations, and interactive elements like questions or exercises where appropriate.`
            }
          ],
          instructionType: 'lessonContentGenerator',
          educationLevel: userEducationLevel
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to load lesson content: ${response.status}`);
      }

      const data = await response.json();
      const responseContent = data.result?.response || data.response || '';
      
      // Update lesson with content
      setLessonPlan(prev => {
        if (!prev) return prev;
        const updated = { ...prev };
        updated.lessons[lessonIndex] = {
          ...updated.lessons[lessonIndex],
          content: responseContent,
          isLoadingContent: false,
          isExpanded: true,
        };
        return updated;
      });

    } catch (err) {
      console.error('Error loading lesson content:', err);
      toast.error('Failed to load lesson content');
      
      // Reset loading state
      setLessonPlan(prev => {
        if (!prev) return prev;
        const updated = { ...prev };
        updated.lessons[lessonIndex] = { ...updated.lessons[lessonIndex], isLoadingContent: false };
        return updated;
      });
    }
  }, [user, getToken, topic, lessonPlan, userEducationLevel]);

  const toggleLessonExpansion = (lessonId: string) => {
    if (!lessonPlan) return;

    const lesson = lessonPlan.lessons.find(l => l.id === lessonId);
    if (!lesson) return;

    if (!lesson.isExpanded) {
      // Expanding the lesson
      if (!lesson.content) {
        // Load content if expanding for the first time
        loadLessonContent(lessonId);
      } else {
        // Content already exists, just expand
        setLessonPlan(prev => {
          if (!prev) return prev;
          const updated = { ...prev };
          const lessonIndex = updated.lessons.findIndex(l => l.id === lessonId);
          if (lessonIndex !== -1) {
            updated.lessons[lessonIndex] = {
              ...updated.lessons[lessonIndex],
              isExpanded: true,
            };
          }
          return updated;
        });
      }
    } else {
      // Collapsing the lesson
      setLessonPlan(prev => {
        if (!prev) return prev;
        const updated = { ...prev };
        const lessonIndex = updated.lessons.findIndex(l => l.id === lessonId);
        if (lessonIndex !== -1) {
          updated.lessons[lessonIndex] = {
            ...updated.lessons[lessonIndex],
            isExpanded: false,
          };
        }
        return updated;
      });
    }
  };

  const removeLesson = (lessonId: string) => {
    setLessonPlan(prev => {
      if (!prev) return prev;
      
      // Prevent deletion if only one lesson remains
      if (prev.lessons.length <= 1) {
        toast.error('Cannot remove the last lesson. Your plan must have at least one lesson.');
        return prev;
      }
      
      return {
        ...prev,
        lessons: prev.lessons.filter(l => l.id !== lessonId),
      };
    });
    toast.success('Lesson removed from plan');
  };

  const regeneratePlan = (mode: 'more' | 'less') => {
    // This would call the API with additional parameters
    toast.info(`Regenerating plan with ${mode} lessons...`);
    generateLessonPlan();
  };

  const saveLessonPlan = useCallback(async () => {
    if (!user || !lessonPlan) {
      toast.error('Please sign in and generate a lesson plan first');
      return;
    }

    setSaving(true);

    try {
      const token = await getToken();
      if (!token) {
        throw new Error('Failed to get authentication token');
      }

      // Prepare the lesson plan data for saving
      const lessonPlanData = {
        topic: topic,
        title: `${topic} - Lesson Plan`,
        totalEstimatedTime: lessonPlan.totalEstimatedTime,
        uuid: generateUserLessonId(), // Generate short ID for the lesson plan

        lessons: lessonPlan.lessons.map((lesson, index) => {
          // If lesson has sections, convert them to content for storage
          let content = lesson.content;
          if (lesson.sections && lesson.sections.length > 0 && !content) {
            content = lesson.sections.map(section => 
              `## ${section.title}\n\n${section.content}`
            ).join('\n\n');
          }
          
          return {
            title: lesson.title,
            description: lesson.description,
            content: content || null,
            uuid: generateUserLessonId(), // Generate short ID for each lesson
            lesson_order: index + 1
          };
        })
      };

      const response = await fetch('/api/d1/user/lesson-plans', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(lessonPlanData),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Failed to save lesson plan:', response.status, errorText);
        throw new Error(`Failed to save lesson plan: ${response.status}`);
      }

      await response.json();
      toast.success('Lesson plan saved successfully!', {
        description: 'You can access it from your dashboard anytime.',
      });
      
      // Close the modal after successful save
      onClose();
    } catch (err) {
      console.error('Error saving lesson plan:', err);
      toast.error('Failed to save lesson plan. Please try again.');
    } finally {
      setSaving(false);
    }
  }, [user, getToken, lessonPlan, topic, onClose]);

  // Handle modal opening logic
  useEffect(() => {
    if (isOpen && topic) {
      if (isTestPrep) {
        // For test prep topics, show preview first
        setShowTestPrepPreview(true);
        setLessonPlan(null);
        setError(null);
        setLoading(false);
      } else {
        // For regular topics, generate AI lesson plan immediately (original behavior)
        generateLessonPlan();
      }
    }
  }, [isOpen, topic, isTestPrep, generateLessonPlan]);

  // Reset state when modal closes
  useEffect(() => {
    if (!isOpen) {
      setLessonPlan(null);
      setError(null);
      setLoading(false);
      setShowTestPrepPreview(false);
    }
  }, [isOpen]);

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-xl">
            <BookOpen className="h-5 w-5" />
            Craft Lesson Plan: {topic}
          </DialogTitle>
          <DialogDescription>
            AI-generated lesson plan with expandable lessons for your selected topic
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto">
          {/* Pre-populated lesson titles preview for test prep */}
          {showTestPrepPreview && (
            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <GraduationCap className="h-5 w-5" />
                    Comprehensive {topic} Study Plan
                  </CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Your personalized lesson plan will include these key topics:
                  </p>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {testPrepLessonPlans.map((plan, index) => (
                      <div key={index} className="flex items-center gap-2 p-2 rounded-lg bg-muted/30">
                        <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center text-xs font-medium text-primary">
                          {index + 1}
                        </div>
                        <span className="text-sm font-medium">{plan.title}</span>
                      </div>
                    ))}
                  </div>
                  <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <p className="text-sm text-blue-800">
                      <strong>Ready to start?</strong> Click "Create All Study Plans" below to generate {testPrepLessonPlans.length} comprehensive 
                      lesson plans (one for each topic above). Each lesson plan will contain 5 individual lessons with 
                      AI-generated content, practice problems, and test-specific strategies.
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {loading && (
            <Card className="h-full flex items-center justify-center">
              <CardContent className="text-center py-8">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
                <h3 className="text-lg font-medium mb-2">Generating Your Lesson Plan</h3>
                <p className="text-muted-foreground">
                  Our AI is crafting a personalized lesson plan for "{topic}"...
                </p>
              </CardContent>
            </Card>
          )}

          {error && (
            <Card className="h-full flex items-center justify-center border-red-200 bg-red-50">
              <CardContent className="text-center py-8">
                <div className="text-red-600 mb-4">
                  <X className="h-8 w-8 mx-auto mb-2" />
                  <h3 className="text-lg font-medium mb-2">Generation Failed</h3>
                  <p className="text-sm">{error}</p>
                </div>
                <Button
                  onClick={generateLessonPlan}
                  variant="outline"
                  className="mt-4"
                >
                  Try Again
                </Button>
                

              </CardContent>
            </Card>
          )}

          {lessonPlan && !loading && (
            <div className="space-y-4">
              {/* Lesson Plan Header */}
              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-lg">Lesson Plan Overview</CardTitle>
                      <p className="text-sm text-muted-foreground mt-1 flex items-center gap-4">
                        <span>{lessonPlan.lessons.length} lessons</span>
                        <span>{lessonPlan.totalEstimatedTime}</span>
                        <span className="flex items-center gap-1">
                          <GraduationCap className="h-3 w-3" />
                          {getEducationLevelDisplayName(userEducationLevel)}
                        </span>
                      </p>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => regeneratePlan('more')}
                        className="text-xs"
                      >
                        <Plus className="h-3 w-3 mr-1" />
                        More Thorough
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => regeneratePlan('less')}
                        className="text-xs"
                      >
                        <Minus className="h-3 w-3 mr-1" />
                        More Succinct
                      </Button>
                    </div>
                  </div>
                </CardHeader>
              </Card>

              {/* Individual Lessons */}
              {lessonPlan.lessons.map((lesson) => (
                <Card key={lesson.id} className="overflow-hidden">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 flex-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => toggleLessonExpansion(lesson.id)}
                          className="p-1 h-auto"
                        >
                          {lesson.isExpanded ? (
                            <ChevronDown className="h-4 w-4" />
                          ) : (
                            <ChevronRight className="h-4 w-4" />
                          )}
                        </Button>
                        <div className="flex-1">
                          <CardTitle className="text-base">{lesson.title}</CardTitle>
                          <p className="text-sm text-muted-foreground mt-1">
                            {lesson.description}
                          </p>
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeLesson(lesson.id)}
                        disabled={lessonPlan.lessons.length <= 1}
                        className={`p-1 h-auto ${
                          lessonPlan.lessons.length <= 1 
                            ? 'text-gray-400 cursor-not-allowed' 
                            : 'text-red-500 hover:text-red-700'
                        }`}
                        title={
                          lessonPlan.lessons.length <= 1 
                            ? "Cannot remove the last lesson. Your plan must have at least one lesson." 
                            : "Remove lesson"
                        }
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardHeader>

                  {lesson.isExpanded && (
                    <CardContent className="pt-0">
                      {lesson.isLoadingContent ? (
                        <div className="flex items-center justify-center py-8">
                          <Loader2 className="h-6 w-6 animate-spin mr-2" />
                          <span className="text-sm text-muted-foreground">Loading lesson content...</span>
                        </div>
                      ) : lesson.sections && lesson.sections.length > 0 ? (
                        <div className="space-y-4">
                          {lesson.sections.map((section, sectionIndex) => (
                            <div key={sectionIndex} className="border rounded-lg p-4 bg-muted/20">
                              <h4 className="font-semibold text-base mb-3">{section.title}</h4>
                              <div className="prose prose-sm max-w-none">
                                <ReactMarkdown>{section.content}</ReactMarkdown>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : lesson.content ? (
                        <div className="border rounded-lg p-4 bg-muted/20">
                          <div className="prose prose-sm max-w-none">
                            <ReactMarkdown>{lesson.content}</ReactMarkdown>
                          </div>
                        </div>
                      ) : (
                        <div className="text-center py-4">
                          <p className="text-sm text-muted-foreground mb-2">Content not loaded yet</p>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => loadLessonContent(lesson.id)}
                          >
                            Load Content
                          </Button>
                        </div>
                      )}
                    </CardContent>
                  )}
                </Card>
              ))}
            </div>
          )}
        </div>

        <div className="flex justify-between items-center pt-4 border-t">
          <div className="flex gap-2">
            {lessonPlan && (
              <Button
                variant="outline"
                onClick={generateLessonPlan}
                size="sm"
                disabled={saving}
              >
                <RefreshCw className="h-4 w-4 mr-1" />
                Regenerate Plan
              </Button>
            )}
          </div>
          <div className="flex gap-2">
            {/* Show Generate Full Plan button for test prep preview */}
            {showTestPrepPreview && (
              <Button 
                onClick={createTestPrepPlans}
                disabled={saving}
                className="min-w-[160px]"
              >
                {saving ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Creating Plans...
                  </>
                ) : (
                  <>
                    <BookOpen className="h-4 w-4 mr-2" />
                    Create All Study Plans
                  </>
                )}
              </Button>
            )}
            {lessonPlan && (
              <Button 
                onClick={saveLessonPlan}
                disabled={saving}
                className="min-w-[120px]"
              >
                {saving ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Saving...
                  </>
                ) : (
                  'Save Plan'
                )}
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default GenerateTopicLessonModal; 