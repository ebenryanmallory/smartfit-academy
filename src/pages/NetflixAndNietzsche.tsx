import React, { useState } from 'react';
import { useUser, SignInButton, SignedOut } from '@clerk/clerk-react';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import GenerateTopicLessonModal from '../components/GenerateTopicLessonModal';
import TaglineComponent from '../components/ui/TaglineComponent';
import { Sparkles, Brain, Users, ArrowRight, Crown, Tv, ScrollText, ChevronLeft, ChevronRight, Clock } from 'lucide-react';

const NetflixAndNietzsche: React.FC = () => {
  const { isSignedIn } = useUser();
  const [isLessonModalOpen, setIsLessonModalOpen] = useState(false);
  const [selectedPairing, setSelectedPairing] = useState<{topic: string, metaTopic: string}>({topic: '', metaTopic: ''});
  const [featuredIndex, setFeaturedIndex] = useState(0);

  const showPairings = [
    {
      show: "Succession",
      text: "Machiavelli's The Prince",
      description: "The Roy family's brutal power dynamics are a masterclass in Machiavellian strategy",
      connection: "Every episode features characters wrestling with whether it's better to be feared or loved, how to maintain power through strategic cruelty, and the art of political manipulation.",
      featuredTitle: "The Roy Playbook Was Written 500 Years Ago",
      featuredDescription: "Every \"Boar on the Floor\" moment. Every calculated betrayal. Every speech about being a killer. The Roy family didn't invent these moves - they perfected what Machiavelli documented centuries ago.",
      featuredNote: "",
      icon: Crown,
      color: "from-amber-500/20 to-orange-500/20",
      textColor: "text-amber-600",
      status: "available"
    },
    {
      show: "The Good Place",
      text: "Multiple Philosophy Texts",
      description: "The show literally teaches moral philosophy through comedy",
      connection: "Features actual discussions of Aristotle's virtue ethics, Kant's categorical imperative, and utilitarian calculations.",
      featuredTitle: "Philosophy Made Fun and Accessible",
      featuredDescription: "Eleanor's journey from selfish to selfless demonstrates virtue ethics in practice. The show makes Aristotle's complex ideas about moral character accessible through comedy and relatable characters.",
      featuredNote: "",
      icon: Sparkles,
      color: "from-green-500/20 to-emerald-500/20",
      textColor: "text-green-600",
      status: "coming-soon"
    },
    {
      show: "Squid Game",
      text: "Rousseau's Social Contract & Marx's Communist Manifesto",
      description: "A hyperbolic representation of capitalism's inequalities",
      connection: "Players literally sign away their rights, the wealthy watch the poor fight for entertainment.",
      featuredTitle: "Capitalism's Dark Mirror Revealed",
      featuredDescription: "The games are a hyperbolic representation of economic inequality. Players sign away their rights, the wealthy watch the poor fight for entertainment - a perfect allegory for class struggle.",
      featuredNote: "",
      icon: Users,
      color: "from-red-500/20 to-pink-500/20",
      textColor: "text-red-600",
      status: "coming-soon"
    },
    {
      show: "Black Mirror",
      text: "Plato's Allegory of the Cave",
      description: "Explores the nature of reality and technology as the new cave wall",
      connection: "Episodes consistently question what's real and what's simulation.",
      featuredTitle: "Technology as the New Cave Wall",
      featuredDescription: "Each episode is essentially a philosophical thought experiment. 'San Junipero,' 'USS Callister,' and 'Bandersnatch' directly question what's real and what's simulation.",
      featuredNote: "",
      icon: Brain,
      color: "from-purple-500/20 to-indigo-500/20",
      textColor: "text-purple-600",
      status: "coming-soon"
    }
  ];

  const handlePairingClick = (pairing: typeof showPairings[0]) => {
    if (pairing.status !== 'available') {
      return;
    }
    
    setSelectedPairing({
      topic: `${pairing.show} × ${pairing.text}`,
      metaTopic: `Netflix & Nietzsche: ${pairing.show}`
    });
    setIsLessonModalOpen(true);
  };

  const handleCloseLessonModal = () => {
    setIsLessonModalOpen(false);
    setSelectedPairing({topic: '', metaTopic: ''});
  };

  const nextFeatured = () => {
    setFeaturedIndex((prev) => (prev + 1) % showPairings.length);
  };

  const prevFeatured = () => {
    setFeaturedIndex((prev) => (prev - 1 + showPairings.length) % showPairings.length);
  };

  const currentPairing = showPairings[featuredIndex];



  return (
    <div className="flex flex-col min-h-screen bg-background">
      {/* Hero Section */}
      <section className="container-section bg-gradient-to-br from-primary/10 via-accent/5 to-secondary/10 py-40">
        <div className="content-container text-center">
          <div className="mb-12">
            <h1 className="text-4xl md:text-6xl font-semibold mb-6 bg-gradient-to-r from-muted via-input to-accent bg-clip-text text-transparent">
              Netflix & Nietzsche
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground mx-auto text-center">
              Your favorite shows aren't just entertainment—they're masterclasses in timeless human nature. 
              From Succession's boardroom betrayals to The Good Place's moral dilemmas, every great story 
              echoes wisdom that philosophers have been teaching for centuries.
            </p>
          </div>

          {/* Featured Pairing Carousel */}
          <Card className={`p-8 backdrop-blur-sm shadow-lg border-2 mx-auto relative ${
            currentPairing.status === 'available' 
              ? 'bg-white/80 border-primary/20' 
              : 'bg-gray-50/80 border-gray-300/30'
          }`}>
            {/* Navigation Arrows */}
            <button
              onClick={prevFeatured}
              className="absolute left-4 top-1/2 transform -translate-y-1/2 p-2 rounded-full bg-white/80 hover:bg-white shadow-md transition-all duration-200 z-10"
              aria-label="Previous pairing"
            >
              <ChevronLeft className="h-5 w-5 text-gray-600" />
            </button>
            <button
              onClick={nextFeatured}
              className="absolute right-4 top-1/2 transform -translate-y-1/2 p-2 rounded-full bg-white/80 hover:bg-white shadow-md transition-all duration-200 z-10"
              aria-label="Next pairing"
            >
              <ChevronRight className="h-5 w-5 text-gray-600" />
            </button>

            <div className="flex flex-col gap-6">
              {currentPairing.status !== 'available' && (
                <div className="text-center">
                  <div className="inline-flex items-center gap-2 bg-gray-500/10 text-gray-600 px-4 py-2 rounded-full text-sm font-medium">
                    <Clock className="h-4 w-4" />
                    Coming Soon
                  </div>
                </div>
              )}
              
              <div className="flex items-center gap-4 justify-center">
                <div className="flex items-center gap-2">
                  <Tv className={`h-8 w-8 ${currentPairing.status === 'available' ? 'text-primary' : 'text-gray-400'}`} />
                  <span className={`text-2xl font-bold ${currentPairing.status === 'available' ? 'text-foreground' : 'text-gray-500'}`}>
                    {currentPairing.show}
                  </span>
                </div>
                <span className={`text-2xl ${currentPairing.status === 'available' ? 'text-muted-foreground' : 'text-gray-400'}`}>×</span>
                <div className="flex items-center gap-2">
                  <ScrollText className={`h-8 w-8 ${currentPairing.status === 'available' ? 'text-accent' : 'text-gray-400'}`} />
                  <span className={`text-2xl font-bold ${currentPairing.status === 'available' ? 'text-foreground' : 'text-gray-500'}`}>
                    {currentPairing.text}
                  </span>
                </div>
              </div>
              
              <div className="text-center">
                <h3 className={`text-xl font-semibold mb-3 ${currentPairing.status === 'available' ? 'text-foreground' : 'text-gray-500'}`}>
                  "{currentPairing.featuredTitle}"
                </h3>
                <p className={`mb-4 max-w-2xl mx-auto ${currentPairing.status === 'available' ? 'text-muted-foreground' : 'text-gray-400'}`}>
                  {currentPairing.featuredDescription}
                </p>

              </div>
              
              <Button 
                onClick={() => handlePairingClick(currentPairing)}
                size="lg"
                className="px-8 mx-auto"
                disabled={currentPairing.status !== 'available'}
              >
                {currentPairing.status === 'available' ? (
                  <>
                    <ArrowRight className="h-4 w-4 mr-2" />
                    Start With Episode 1 × Chapter 1
                  </>
                ) : (
                  <>
                    <Clock className="h-4 w-4 mr-2" />
                    Coming Soon
                  </>
                )}
              </Button>

              {/* Pagination Dots */}
              <div className="flex justify-center gap-2 mt-2">
                {showPairings.map((pairing, index) => (
                  <button
                    key={index}
                    onClick={() => setFeaturedIndex(index)}
                    className={`w-2 h-2 rounded-full transition-all duration-200 ${
                      index === featuredIndex 
                        ? pairing.status === 'available' 
                          ? 'bg-primary w-8' 
                          : 'bg-gray-400 w-8'
                        : pairing.status === 'available'
                          ? 'bg-gray-300 hover:bg-gray-400'
                          : 'bg-gray-200 hover:bg-gray-300'
                    }`}
                    aria-label={`Go to ${showPairings[index].show} pairing${pairing.status !== 'available' ? ' (Coming Soon)' : ''}`}
                  />
                ))}
              </div>
              
              <SignedOut>
                <p className="text-xs text-primary font-medium text-center bg-primary/5 p-3 rounded">
                  ✨ Preview lessons for free! Sign up to save your progress and unlock all pairings
                </p>
              </SignedOut>
            </div>
          </Card>

          {/* Tagline */}
          <TaglineComponent className="mt-8 mx-auto" />
        </div>
      </section>

      {/* All Pairings Grid */}
      <section className="container-section bg-gradient-to-r from-accent/5 to-primary/5" data-pairings-section>
        <div className="content-container">
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-foreground">
            Philosophy in Action
          </h2>
          <p className="text-lg text-muted-foreground mb-12 mx-auto">
            See how your favorite characters navigate the same moral challenges that have fascinated thinkers for millennia. 
            Each connection reveals the philosophical foundations behind compelling storytelling.
          </p>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-2 gap-6 mx-auto">
            {showPairings.map((pairing, index) => {
              const IconComponent = pairing.icon;
              const isComingSoon = pairing.status !== 'available';
              
              return (
                <Card 
                  key={index} 
                  className={`p-6 transition-all duration-300 border backdrop-blur-sm relative
                    ${isComingSoon 
                      ? 'opacity-60 cursor-not-allowed bg-gradient-to-br from-gray-100/50 to-gray-200/50 hover:opacity-70' 
                      : `hover:shadow-lg cursor-pointer hover:border-primary/50 bg-palette-3`
                    }`}
                  onClick={() => handlePairingClick(pairing)}
                >
                  {isComingSoon && (
                    <div className="absolute top-3 right-3 bg-gray-500/90 text-white text-xs px-2 py-1 rounded-full flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      Coming Soon
                    </div>
                  )}
                  
                  <div className="flex flex-col h-full">
                    <div className="flex items-center gap-3 mb-4">
                      <div className={`p-2 rounded-lg ${isComingSoon ? 'bg-gray-200/80 text-gray-400' : `bg-white/80 ${pairing.textColor}`}`}>
                        <IconComponent className="h-6 w-6" />
                      </div>
                      <div className="flex-1">
                        <h3 className={`font-bold text-lg ${isComingSoon ? 'text-gray-500' : 'text-foreground'}`}>
                          {pairing.show}
                        </h3>
                        <p className={`text-sm ${isComingSoon ? 'text-gray-400' : 'text-muted-foreground'}`}>
                          {pairing.text}
                        </p>
                      </div>
                    </div>
                    
                    <p className={`text-sm font-medium mb-2 ${isComingSoon ? 'text-gray-500' : 'text-foreground'}`}>
                      {pairing.description}
                    </p>
                    
                    <p className={`text-xs mb-4 flex-1 ${isComingSoon ? 'text-gray-400' : 'text-muted-foreground'}`}>
                      {pairing.connection}
                    </p>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1">
                      </div>
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        className={`text-xs ${isComingSoon ? 'text-gray-400 cursor-not-allowed' : ''}`}
                        disabled={isComingSoon}
                      >
                        {isComingSoon ? 'Coming Soon' : 'Explore →'}
                      </Button>
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        </div>
      </section>

      {/* Quote Section */}
      <section className="container-section" style={{ background: 'linear-gradient(to top right, var(--color-accent), var(--color-background))', textShadow: '2px 3px 3px rgba(75, 75, 75, .4)'  }}>
        <div className="content-container text-center py-16">
          <div className="text-left w-full text-8xl md:text-9xl text-white leading-none mb-4" style={{ fontFamily: 'cursive' }}>"</div>
          <blockquote className="text-2xl md:text-4xl font-light text-white leading-relaxed mb-6">
            History never repeats itself. Man always does.
          </blockquote>
          <cite className="text-lg md:text-xl text-white/80 font-medium">
            — Voltaire
          </cite>
          <div className="text-justify w-full text-8xl md:text-9xl text-white leading-none mt-4 rotate-180" style={{ fontFamily: 'cursive' }}>"</div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="container-section bg-gradient-to-br from-orange-50/50 via-amber-50/30 to-yellow-50/50">
        <div className="content-container">
          <div className="mb-12 w-full md:w-2/3">
            <h2 className="text-3xl md:text-4xl font-bold mb-6 text-foreground">
              Your Cozy Challenge Awaits
            </h2>
            <p className="text-lg text-muted-foreground mx-auto">
              Transform your next movie night into something special. Snuggle up with your favorite show 
              and discover the timeless wisdom hidden in every scene.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <Card className="p-6 bg-white/80 backdrop-blur-sm border border-orange-200/50 hover:shadow-lg transition-all duration-300">
              <div className="text-center">
                <div className="w-16 h-16 bg-[#c7522a] rounded-full flex items-center justify-center mx-auto mb-4">
                  <Tv className="h-8 w-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3 text-foreground">Pick Your Show</h3>
                <p className="text-muted-foreground text-sm">
                  Choose from our curated pairings or suggest your own favorite series. 
                  Each show comes with a custom lesson plan designed just for cozy exploration.
                </p>
              </div>
            </Card>

            <Card className="p-6 bg-white/80 backdrop-blur-sm border border-orange-200/50 hover:shadow-lg transition-all duration-300">
              <div className="text-center">
                <div className="w-16 h-16 bg-[#c7522a] rounded-full flex items-center justify-center mx-auto mb-4">
                  <ScrollText className="h-8 w-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3 text-foreground">Follow Your Lesson</h3>
                <p className="text-muted-foreground text-sm">
                  Start shallow with quick parallels to classic thinking, or dive deep into the original texts. 
                  Your lesson plan adapts to how curious you're feeling tonight.
                </p>
              </div>
            </Card>

            <Card className="p-6 bg-white/80 backdrop-blur-sm border border-orange-200/50 hover:shadow-lg transition-all duration-300">
              <div className="text-center">
                <div className="w-16 h-16 bg-[#c7522a] rounded-full flex items-center justify-center mx-auto mb-4">
                  <Brain className="h-8 w-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3 text-foreground">Discover & Connect</h3>
                <p className="text-muted-foreground text-sm">
                  Watch familiar characters make choices that echo through centuries of human thought. 
                  See how every great story connects to timeless wisdom.
                </p>
              </div>
            </Card>
          </div>

          <div className="mt-12 bg-gradient-to-r from-orange-100/50 to-amber-100/50 rounded-2xl p-8 border border-orange-200/30">
            <div className="text-center">
              <h3 className="text-2xl font-semibold mb-4 text-foreground">Your Challenge</h3>
              <p className="text-lg text-muted-foreground max-w-4xl mx-auto leading-relaxed">
                Next time you're settling in for a cozy evening with your favorite show, grab a warm drink 
                and open your lesson plan. Whether you want to spend just 5 minutes connecting the dots 
                or an hour diving into the philosophical depths, you'll never watch the same way again. 
                Every episode becomes a window into the great conversations humanity has been having for centuries.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="bg-palette-2">
        <div className="container-section content-container text-center">
          <h2 className="text-3xl font-bold mb-6">Ready to See Your Shows Differently?</h2>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Transform your binge-watching into genuine wisdom. Every great story has been told before - 
            learn from the masters who told it first.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button 
              className="bg-primary text-white hover:bg-primary/90"
              size="lg" 
              onClick={() => {
                const pairingsSection = document.querySelector('[data-pairings-section]');
                if (pairingsSection) {
                  pairingsSection.scrollIntoView({ behavior: 'smooth' });
                } else {
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }
              }}
            >
              Explore
            </Button>
            <SignedOut>
              <SignInButton mode="modal">
                <Button 
                  size="lg" 
                  variant="outline"
                  className="border-primary text-primary hover:bg-primary/10"
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
        topic={selectedPairing.topic}
        useRelevanceEngine={true}
        previewMode={!isSignedIn}
        metaTopic={selectedPairing.metaTopic}
      />
    </div>
  );
};

export default NetflixAndNietzsche; 