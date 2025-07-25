import { Link } from 'react-router-dom';
import { useState, useRef } from 'react';
import { Button } from '../components/ui/button';
import BottomChatAssistant from "../components/BottomChatAssistant";
import UserTopics, { UserTopicsRef } from "../components/UserTopics";
import SavedLessonPlans, { SavedLessonPlansRef } from "../components/SavedLessonPlans";
import GenerateTopicLessonModal from "../components/GenerateTopicLessonModal";
import CreateYourJourney from "../components/CreateYourJourney";
import { useUser } from '@clerk/clerk-react';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '../components/ui/card';
import { GraduationCap, Zap, Users, CheckCircle } from "lucide-react";

function Dashboard() {
  const [isAssistantExpanded, setIsAssistantExpanded] = useState(false);
  const [isLessonModalOpen, setIsLessonModalOpen] = useState(false);
  const [selectedTopic, setSelectedTopic] = useState<string>('');
  const [selectedMetaTopic, setSelectedMetaTopic] = useState<string>('');
  const [useTestPrepMode, setUseTestPrepMode] = useState(false);
  const [hasLessonPlans, setHasLessonPlans] = useState(false);
  const [userLatestInput, setUserLatestInput] = useState<string>('');
  const { isSignedIn } = useUser();
  const userTopicsRef = useRef<UserTopicsRef>(null);
  const userLessonPlansRef = useRef<SavedLessonPlansRef>(null);

  const handleExpandAssistant = () => {
    setIsAssistantExpanded(true);
    setUseTestPrepMode(false); // Ensure test prep mode is off when using assistant
    // Apply blur to main content only
    const mainContent = document.getElementById('main-dashboard-content');
    if (mainContent) {
      mainContent.style.filter = 'blur(8px)';
      mainContent.style.transition = 'filter 0.3s ease-in-out';
    }
  };

  const handleTopicClick = (topic: string) => {
    setSelectedTopic(topic);
    setSelectedMetaTopic(''); // Clear meta topic for regular topics
    setUseTestPrepMode(false); // Regular topics from user's saved topics
    setIsLessonModalOpen(true);
  };

  const handleTopicSaved = async () => {
    // Refresh the topics when a new topic is saved
    await userTopicsRef.current?.refreshTopics();
    // Also refresh lesson plans in case a new lesson plan was generated
    await userLessonPlansRef.current?.refreshLessonPlans();
  };

  const handleLessonPlansChange = (lessonPlans: any[]) => {
    setHasLessonPlans(lessonPlans.length > 0);
  };

  const handleToggleAssistant = () => {
    const newExpandedState = !isAssistantExpanded;
    setIsAssistantExpanded(newExpandedState);
    
    // Apply or remove blur based on the new state
    const mainContent = document.getElementById('main-dashboard-content');
    if (mainContent) {
      if (newExpandedState) {
        mainContent.style.filter = 'blur(8px)';
        mainContent.style.transition = 'filter 0.3s ease-in-out';
      } else {
        mainContent.style.filter = 'none';
        mainContent.style.transition = 'filter 0.3s ease-in-out';
      }
    }
  };

  const handleCloseLessonModal = () => {
    setIsLessonModalOpen(false);
    setSelectedTopic('');
    setSelectedMetaTopic('');
    setUseTestPrepMode(false);
  };

  const handleUserInput = (userInput: string) => {
    setUserLatestInput(userInput);
  };

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <div id="main-dashboard-content" className="pb-chat-assistant">
      {/* User's Lesson Plans - Show if user is signed in and has saved lesson plans */}
      {isSignedIn && (
        <div className="content-container my-12 w-full">
          <SavedLessonPlans 
            ref={userLessonPlansRef}
            onLessonPlansChange={handleLessonPlansChange}
            className="mb-8"
          />
        </div>
      )}

      {/* Hero/Intro Section - Only show if user is not signed in OR has no lesson plans */}
      {(!isSignedIn || !hasLessonPlans) && (
        <section className="container-section">
          <div className="content-container-md">
            <p className="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto text-center">
              A next-generation learning platform that leverages AI to deliver adaptive, personalized education from elementary through graduate levels. Built for modern learners, powered by cutting-edge technology, and designed for your success.
            </p>
            <div className="flex flex-wrap justify-center gap-4 mt-8">
              <Button variant="secondary" asChild className='button-padding'>
                <Link to="/dashboard/lessons">Available lessons</Link>
              </Button>
                              <Button variant="outline" asChild className='button-padding'>
                  <Link to="/lessons/c-intro-ai">Try a Sample Lesson</Link>
                </Button>

            </div>
          </div>
        </section>
      )}

      {/* User's Learning Topics - Show prominently if user has saved topics */}
      {isSignedIn && (
        <UserTopics 
          ref={userTopicsRef}
          onTopicClick={handleTopicClick}
          className="mb-8"
          userInputTopic={userLatestInput}
        />
      )}

      {/* Features Overview - Only show for non-signed-in users */}
      {!isSignedIn && (
        <section className="container-section">
          <div className="content-container">
            <h2 className="text-3xl md:text-4xl font-bold text-center mb-10 text-foreground">Platform Features</h2>
            <div className="responsive-grid">
              <Card className="feature-card">
                <CardHeader>
                  <div className="mb-4">
                    <GraduationCap className="h-8 w-8 text-primary mx-auto" />
                  </div>
                  <CardTitle className="text-xl font-bold mb-2">Onboarding & Assessment</CardTitle>
                  <CardDescription>Skill quiz to place learners at the right level</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Take a quick skill assessment to personalize your journey and start at the level that's right for you.
                  </p>
                </CardContent>
              </Card>

              <Card className="feature-card">
                <CardHeader>
                  <div className="mb-4">
                    <Zap className="h-8 w-8 text-accent mx-auto" />
                  </div>
                  <CardTitle className="text-xl font-bold mb-2">Adaptive Recommendations</CardTitle>
                  <CardDescription>Claude-powered lesson suggestions</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Receive personalized lesson and project recommendations, powered by AI, to maximize your learning efficiency.
                  </p>
                </CardContent>
              </Card>

              <Card className="feature-card">
                <CardHeader>
                  <div className="mb-4">
                    <Users className="h-8 w-8 text-secondary mx-auto" />
                  </div>
                  <CardTitle className="text-xl font-bold mb-2">Interactive Lesson Viewer</CardTitle>
                  <CardDescription>Code sandboxes & live feedback</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Explore lessons with embedded code playgrounds and instant feedback to reinforce concepts as you learn.
                  </p>
                </CardContent>
                                  <CardFooter className="bg-muted/20 pt-0">
                    <Button asChild variant="outline" size="sm" className="w-full button-padding">
                      <Link to="/lessons/c-intro-ai">Try a Lesson</Link>
                    </Button>
                  </CardFooter>
              </Card>

              <Card className="feature-card">
                <CardHeader>
                  <div className="mb-4">
                    <CheckCircle className="h-8 w-8 text-success mx-auto" />
                  </div>
                  <CardTitle className="text-xl font-bold mb-2">Chat Tutor & Progress Dashboard</CardTitle>
                  <CardDescription>AI Q&A and achievement tracking</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Get instant help from our AI-powered tutor and track your progress, achievements, and badges as you advance.
                  </p>
                </CardContent>
                <CardFooter className="bg-muted/20 pt-0">
                  <Button asChild variant="outline" size="sm" className="w-full button-padding">
                    <Link to="/dashboard/lessons">Go to Dashboard</Link>
                  </Button>
                </CardFooter>
              </Card>
            </div>
          </div>
        </section>
      )}

      {/* Create Your Journey Section */}
      <CreateYourJourney onExpandAssistant={handleExpandAssistant} />

      {/* Create Topics by Goal Section */}
      <section className="container-section">
        <div className="content-container text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6 text-foreground">
            Create Topics by Goal
          </h2>
          <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto">
            Preparing for a standardized test? Let our AI create a comprehensive study plan tailored to your target exam. 
            Get structured topics, practice materials, and a personalized timeline to help you achieve your best score.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Button 
              size="lg" 
              variant="outline"
              className="text-lg px-8 py-4 h-auto"
              onClick={() => {
                setSelectedTopic('GED Test Preparation');
                setSelectedMetaTopic('GED');
                setUseTestPrepMode(true);
                setIsLessonModalOpen(true);
              }}
            >
              GED Prep
            </Button>
            <Button 
              size="lg" 
              variant="outline"
              className="text-lg px-8 py-4 h-auto"
              onClick={() => {
                setSelectedTopic('SAT Test Preparation');
                setSelectedMetaTopic('SAT');
                setUseTestPrepMode(true);
                setIsLessonModalOpen(true);
              }}
            >
              SAT Prep
            </Button>
            <Button 
              size="lg" 
              variant="outline"
              className="text-lg px-8 py-4 h-auto"
              onClick={() => {
                setSelectedTopic('ACT Test Preparation');
                setSelectedMetaTopic('ACT');
                setUseTestPrepMode(true);
                setIsLessonModalOpen(true);
              }}
            >
              ACT Prep
            </Button>
          </div>
        </div>
      </section>
      
      {/* Value Proposition & How It Works */}
      <section className="container-section bg-secondary">
        <div className="content-container">
          <h2 className="text-3xl font-bold text-center mb-12 text-foreground">Why SmartFit?</h2>
          <div className="two-column-grid">
            <Card className="info-card">
              <CardHeader className="px-0 pt-0">
                <CardTitle className="text-xl font-semibold text-foreground">Personalized, Adaptive Learning</CardTitle>
              </CardHeader>
              <CardContent className="px-0 pb-0">
                <p className="text-muted-foreground">
                  Our platform adapts to your skill level, learning style, and pace. Every lesson and recommendation is tailored to help you succeed.
                </p>
              </CardContent>
            </Card>

            <Card className="info-card">
              <CardHeader className="px-0 pt-0">
                <CardTitle className="text-xl font-semibold text-foreground">Modern, Engaging Experience</CardTitle>
              </CardHeader>
              <CardContent className="px-0 pb-0">
                <p className="text-muted-foreground">
                  Enjoy interactive lessons, instant feedback, and a beautiful, distraction-free interface designed for real progress.
                </p>
              </CardContent>
            </Card>

            <Card className="info-card">
              <CardHeader className="px-0 pt-0">
                <CardTitle className="text-xl font-semibold text-foreground">Powered by Leading AI</CardTitle>
              </CardHeader>
              <CardContent className="px-0 pb-0">
                <p className="text-muted-foreground">
                  Built on Anthropic Claude and Cloudflare's edge network, our AI tutor and adaptive engine provide world-class support and recommendations.
                </p>
              </CardContent>
            </Card>

            <Card className="info-card">
              <CardHeader className="px-0 pt-0">
                <CardTitle className="text-xl font-semibold text-foreground">Track & Celebrate Your Growth</CardTitle>
              </CardHeader>
              <CardContent className="px-0 pb-0">
                <p className="text-muted-foreground">
                  Visualize your journey, unlock badges, and see your skills grow with our progress dashboard.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="container-section bg-palette-2">
        <div className="content-container text-center">
          <h2 className="text-3xl font-bold mb-6 text-foreground">Ready to Transform Your Learning?</h2>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Join thousands of learners who are already advancing their education with AI-powered personalized learning.
          </p>
          <Button className="btn-primary" asChild>
            <Link to="/onboarding">Start Your Journey</Link>
          </Button>
        </div>
      </section>
      </div>

      <BottomChatAssistant 
        isExpanded={isAssistantExpanded} 
        onToggleExpanded={handleToggleAssistant}
        onTopicSaved={handleTopicSaved}
        onUserInput={handleUserInput}
      />

      <GenerateTopicLessonModal
        isOpen={isLessonModalOpen}
        onClose={handleCloseLessonModal}
        topic={selectedTopic}
        useTestPrep={useTestPrepMode}
        metaTopic={selectedMetaTopic}
      />
    </div>
  );
}

export default Dashboard;