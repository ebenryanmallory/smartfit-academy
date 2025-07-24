import React, { useState } from 'react';
import { useUser, SignInButton, SignedIn, SignedOut } from '@clerk/clerk-react';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { Input } from '../components/ui/input';
import GenerateTopicLessonModal from '../components/GenerateTopicLessonModal';
import HistoricalConnectionSummary from '../components/HistoricalConnectionSummary';
import TaglineComponent from '../components/ui/TaglineComponent';
import { Sparkles, History, TrendingUp, Brain, Zap, BookOpen, Users, ArrowRight } from 'lucide-react';

const RelevanceEngine: React.FC = () => {
  const { isSignedIn } = useUser();
  const [topicInput, setTopicInput] = useState('');
  const [isLessonModalOpen, setIsLessonModalOpen] = useState(false);
  const [selectedTopic, setSelectedTopic] = useState<string>('');
  const [exploredTopic, setExploredTopic] = useState<string>('');

  // Pre-built trending topics with examples
  const trendingTopics = [
    {
      topic: "Bitcoin and cryptocurrency craze",
      description: "Connect to economic philosophies of Adam Smith and John Maynard Keynes",
      icon: TrendingUp
    },
    {
      topic: "Social media cancel culture",
      description: "Explore Ancient Roman mob mentality and its consequences",
      icon: Users
    },
    {
      topic: "AI replacing human jobs",
      description: "Learn from the Industrial Revolution and Luddite movement",
      icon: Brain
    },
    {
      topic: "Climate change activism",
      description: "Study environmental movements throughout history",
      icon: Sparkles
    },
    {
      topic: "Political polarization online",
      description: "Understand civil discourse from Ancient Greek democracy",
      icon: History
    },
    {
      topic: "Influencer marketing economy",
      description: "Connect to patronage systems in Renaissance art",
      icon: Zap
    }
  ];

  const handleTopicSubmit = (topic: string) => {
    if (!topic.trim()) return;
    
    setExploredTopic(topic.trim());
    
  };

  const handlePrebuiltTopicClick = (topic: string) => {
    setTopicInput(topic);
  };

  const handleCloseLessonModal = () => {
    setIsLessonModalOpen(false);
    setSelectedTopic('');
  };

  const handleGenerateLesson = (lessonTopic: string) => {
    setSelectedTopic(lessonTopic);
    setIsLessonModalOpen(true);
  };

  return (
    <div className="flex flex-col min-h-screen bg-background">
      {/* Hero Section - Topic Input Above the Fold */}
      <section className="container-section bg-gradient-to-br from-primary/10 via-accent/5 to-secondary/10 py-40">
        <div className="content-container text-center">
          {/* Campaign Title */}
          <div className="mb-12">
            <p className="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto text-center">
              Connect any trending topic to timeless wisdom. Our AI instantly bridges today's conversations 
              with classical texts and historical insights.
            </p>
          </div>

          {/* Topic Input - Above the Fold */}
          <Card className="p-6 bg-white/80 backdrop-blur-sm shadow-lg border-2 border-primary/20">
            <div className="flex flex-col gap-4">
              <div className="flex items-center gap-2 text-left">
                <h3 className="text-lg font-semibold text-foreground">
                  What's trending?
                </h3>
              </div>
              <div className="flex gap-2">
                <Input
                  placeholder="Enter any social media trend, news topic, or cultural phenomenon..."
                  value={topicInput}
                  onChange={(e) => setTopicInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleTopicSubmit(topicInput);
                    }
                  }}
                  className="flex-1 text-base h-12"
                />
                <Button 
                  onClick={() => handleTopicSubmit(topicInput)}
                  size="lg"
                  className="px-6"
                  disabled={!topicInput.trim()}
                >
                  <ArrowRight className="h-4 w-4 ml-1" />
                  Explore
                </Button>
              </div>
              <p className="text-sm text-muted-foreground text-left">
                Examples: "NFT art bubble", "TikTok dance trends", "Crypto Twitter drama", "Meme stock trading"
              </p>
              <SignedOut>
                <p className="text-xs text-primary font-medium text-left bg-primary/5 p-2 rounded">
                  ✨ Try it now for free! No sign-up required to preview lessons
                </p>
              </SignedOut>
            </div>
          </Card>

            {/* Quick Topic Options */}
            <div className="max-w-4xl mx-auto mt-6">
              <p className="text-sm text-muted-foreground text-center mb-3">
                Or click any trending topic to add it to your search:
              </p>
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
                {trendingTopics.map((item, index) => {
                  const IconComponent = item.icon;
                  return (
                    <Card 
                      key={index} 
                      className="p-2 hover:shadow-md transition-all duration-200 cursor-pointer border hover:border-primary/50 bg-white/80 backdrop-blur-sm"
                      onClick={() => handlePrebuiltTopicClick(item.topic)}
                    >
                      <div className="flex flex-col items-center text-center">
                        <div className="p-1 rounded bg-primary/10 mb-1">
                          <IconComponent className="h-3 w-3 text-primary" />
                        </div>
                        <p className="text-xs font-medium text-foreground leading-tight mb-1">
                          {item.topic}
                        </p>
                        <Button variant="ghost" size="sm" className="text-xs h-5 px-2 py-0">
                          Add
                        </Button>
                      </div>
                    </Card>
                  );
                })}
              </div>
          </div>

          {/* Tagline */}
          <TaglineComponent className="mt-8 mx-auto" />

          {/* Historical Connection Summary */}
          {exploredTopic && (
            <div data-connection-summary>
              <HistoricalConnectionSummary
                topic={exploredTopic}
                onGenerateLesson={handleGenerateLesson}
                className="mb-8"
              />
            </div>
          )}

        </div>
      </section>


      {/* Inspirational Quote */}
      <section className="container-section" style={{ background: 'linear-gradient(to top right, var(--color-accent), var(--color-background))', textShadow: '2px 3px 3px rgba(75, 75, 75, .4)'  }}>
        <div className="content-container text-center py-16">
          <div className="text-left w-full text-8xl md:text-9xl text-white leading-none mb-4" style={{ fontFamily: 'cursive' }}>"</div>
          <blockquote className="text-2xl md:text-4xl font-light text-white leading-relaxed mb-6">
            What has been will be again, what has been done will be done again; 
            there is nothing new under the sun.
          </blockquote>
          <cite className="text-lg md:text-xl text-white/80 font-medium">
            — Ecclesiastes 1:9
          </cite>
          <div className="text-justify w-full text-8xl md:text-9xl text-white leading-none mt-4 rotate-180" style={{ fontFamily: 'cursive' }}>"</div>
        </div>
      </section>

      {/* Value Proposition */}
      <section className="container-section bg-gradient-to-r from-accent/5 to-primary/5">
        <div className="content-container">
          <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
            <Card className="p-4 bg-white/60 backdrop-blur-sm">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="h-5 w-5 text-accent" />
                <h4 className="font-semibold">Instant Connections</h4>
              </div>
              <p className="text-sm text-muted-foreground">
                AI finds surprising links between today's trends and historical events
              </p>
            </Card>
            <Card className="p-4 bg-white/60 backdrop-blur-sm">
              <div className="flex items-center gap-2 mb-2">
                <BookOpen className="h-5 w-5 text-primary" />
                <h4 className="font-semibold">Deep Understanding</h4>
              </div>
              <p className="text-sm text-muted-foreground">
                Go beyond surface-level takes with grounded historical context
              </p>
            </Card>
            <Card className="p-4 bg-white/60 backdrop-blur-sm">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="h-5 w-5 text-secondary" />
                <h4 className="font-semibold">Applied Philosophy</h4>
              </div>
              <p className="text-sm text-muted-foreground">
                Discover how classical thinkers already solved today's "new" problems
              </p>
            </Card>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="container-section bg-gradient-to-r from-accent/10 to-primary/10">
        <div className="content-container">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-10 text-foreground">
            How The Relevance Engine Works
          </h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-primary">1</span>
              </div>
              <h3 className="text-xl font-semibold mb-3">You Share What's Trending</h3>
              <p className="text-muted-foreground">
                Input any current topic, trend, or controversy you've seen online or in the news.
              </p>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-accent/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-accent">2</span>
              </div>
              <h3 className="text-xl font-semibold mb-3">AI Finds Historical Parallels</h3>
              <p className="text-muted-foreground">
                Our engine analyzes the topic and identifies relevant historical events, thinkers, and texts.
              </p>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-secondary/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-secondary">3</span>
              </div>
              <h3 className="text-xl font-semibold mb-3">You Get Custom Lessons</h3>
              <p className="text-muted-foreground">
                Receive personalized learning paths that connect modern issues to timeless wisdom.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Examples Section */}
      <section className="container-section">
        <div className="content-container">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-10 text-foreground">
            Real Examples
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <Card className="p-6 bg-gradient-to-br from-primary/5 to-accent/5">
              <h3 className="text-xl font-semibold mb-3 text-foreground">
                "Why is everyone obsessed with crypto?"
              </h3>
              <p className="text-muted-foreground mb-4">
                <strong>Historical Connection:</strong> Tulip mania in 17th century Netherlands, 
                the California Gold Rush, and economic bubble theories from Adam Smith to Keynes.
              </p>
              <div className="flex items-center gap-2 text-sm text-primary">
                <History className="h-4 w-4" />
                <span>Connects to: Economics, Psychology, History</span>
              </div>
            </Card>
            
            <Card className="p-6 bg-gradient-to-br from-secondary/5 to-primary/5">
              <h3 className="text-xl font-semibold mb-3 text-foreground">
                "Cancel culture is getting out of hand"
              </h3>
              <p className="text-muted-foreground mb-4">
                <strong>Historical Connection:</strong> Ancient Roman ostracism, Salem witch trials, 
                and philosophical debates on justice from Aristotle to John Stuart Mill.
              </p>
              <div className="flex items-center gap-2 text-sm text-secondary">
                <BookOpen className="h-4 w-4" />
                <span>Connects to: Philosophy, Sociology, Political Science</span>
              </div>
            </Card>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="container-section" style={{ background: 'linear-gradient(to bottom right, var(--color-secondary), var(--color-background))', textShadow: '1px 2px 3px rgba(75, 75, 75, .2)'  }}>
        <div className="content-container text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6 text-white">
            Ready to Travel Through Time?
          </h2>
          <p className="text-xl text-white/90 mb-8 max-w-2xl mx-auto">
            Transform your curiosity about current events into deep, lasting knowledge 
            with connections to history's greatest minds.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button 
              size="lg" 
              variant="outline"
              className="bg-white border-white hover:bg-white/90"
              onClick={() => {
                const topicSection = document.querySelector('.container-section');
                topicSection?.scrollIntoView({ behavior: 'smooth' });
              }}
            >
              Start Exploring Now
            </Button>
            <SignedOut>
              <SignInButton mode="modal">
                <Button 
                  size="lg" 
                  variant="outline"
                  className="bg-transparent text-white border-white hover:bg-white/10"
                >
                  Sign Up for Free
                </Button>
              </SignInButton>
            </SignedOut>
          </div>
        </div>
      </section>

      {/* Modal for generating lessons */}
      <GenerateTopicLessonModal
        isOpen={isLessonModalOpen}
        onClose={handleCloseLessonModal}
        topic={selectedTopic}
        useRelevanceEngine={true}
        previewMode={!isSignedIn}
        metaTopic={exploredTopic}
      />
    </div>
  );
};

export default RelevanceEngine; 