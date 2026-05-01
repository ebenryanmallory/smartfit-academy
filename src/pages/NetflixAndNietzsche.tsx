import React, { useState } from 'react';
import { useUser, SignInButton, SignedOut } from '@clerk/react';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import GenerateTopicLessonModal from '../components/GenerateTopicLessonModal';
import VideoModal from '../components/VideoModal';
import TaglineComponent from '../components/ui/TaglineComponent';
import ShowPairingCarousel from '../components/ShowPairingCarousel';
import { showPairings, ShowPairing } from '../data/showPairings';
import { Brain, Clock, Tv, ScrollText } from 'lucide-react';

const NetflixAndNietzsche: React.FC = () => {
  const { isSignedIn } = useUser();
  const [isLessonModalOpen, setIsLessonModalOpen] = useState(false);
  const [selectedPairing, setSelectedPairing] = useState<{topic: string, metaTopic: string}>({topic: '', metaTopic: ''});
  const [isVideoModalOpen, setIsVideoModalOpen] = useState(false);
  const [currentVideoSrc, setCurrentVideoSrc] = useState('');

  const handlePairingClick = (pairing: ShowPairing) => {
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

  const handleVideoClick = (videoSrc: string) => {
    setCurrentVideoSrc(videoSrc);
    setIsVideoModalOpen(true);
  };

  const handleCloseVideoModal = () => {
    setIsVideoModalOpen(false);
    setCurrentVideoSrc('');
  };

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
          <ShowPairingCarousel
            pairings={showPairings}
            onPairingClick={handlePairingClick}
            onVideoClick={handleVideoClick}
            showSignedOutPrompt={true}
          />

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

      {/* Video Modal */}
      <VideoModal
        isOpen={isVideoModalOpen}
        onClose={handleCloseVideoModal}
        videoSrc={currentVideoSrc}
        title="Succession Promo"
      />
    </div>
  );
};

export default NetflixAndNietzsche;
