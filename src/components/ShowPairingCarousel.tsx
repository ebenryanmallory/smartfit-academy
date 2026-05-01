import React, { useState } from 'react';
import { SignedOut } from '@clerk/react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { ArrowRight, Tv, ScrollText, ChevronLeft, ChevronRight, Clock, BookOpen, Play, ExternalLink } from 'lucide-react';
import { ShowPairing } from '../data/showPairings';

interface ShowPairingCarouselProps {
  pairings: ShowPairing[];
  onPairingClick: (pairing: ShowPairing) => void;
  onVideoClick?: (videoSrc: string) => void;
  showSignedOutPrompt?: boolean;
}

const ShowPairingCarousel: React.FC<ShowPairingCarouselProps> = ({
  pairings,
  onPairingClick,
  onVideoClick,
  showSignedOutPrompt = true,
}) => {
  const [featuredIndex, setFeaturedIndex] = useState(0);

  const currentPairing = pairings[featuredIndex];

  const next = () => setFeaturedIndex((prev) => (prev + 1) % pairings.length);
  const prev = () => setFeaturedIndex((prev) => (prev - 1 + pairings.length) % pairings.length);

  return (
    <Card className={`p-8 backdrop-blur-sm shadow-lg border-2 mx-auto relative ${
      currentPairing.status === 'available'
        ? 'bg-white/80 border-primary/20'
        : 'bg-gray-50/80 border-gray-300/30'
    }`}>
      {/* Navigation Arrows */}
      <button
        onClick={prev}
        className="absolute left-4 top-1/2 transform -translate-y-1/2 p-2 rounded-full bg-white/80 hover:bg-white shadow-md transition-all duration-200 z-10"
        aria-label="Previous pairing"
      >
        <ChevronLeft className="h-5 w-5 text-gray-600" />
      </button>
      <button
        onClick={next}
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
            {currentPairing.featuredTitle}
          </h3>
          <p className={`mb-4 max-w-2xl mx-auto ${currentPairing.status === 'available' ? 'text-muted-foreground' : 'text-gray-400'}`}>
            {currentPairing.featuredDescription}
          </p>

          <div className="flex flex-wrap justify-center gap-3 mb-4">
            {currentPairing.show === 'Succession' && currentPairing.status === 'available' && currentPairing.bookUrl && (
              <a
                href={currentPairing.bookUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 bg-amber-50 hover:bg-amber-100 text-amber-700 rounded-lg text-sm font-medium transition-colors"
              >
                <BookOpen className="h-4 w-4" />
                Read The Prince
              </a>
            )}

            {currentPairing.show === 'Succession' && currentPairing.status === 'available' && currentPairing.promoVideoUrl && onVideoClick && (
              <button
                onClick={() => onVideoClick(currentPairing.promoVideoUrl!)}
                className="inline-flex items-center gap-2 px-4 py-2 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-lg text-sm font-medium transition-colors"
              >
                <Play className="h-4 w-4" />
                15s Promo
              </button>
            )}

            {currentPairing.seriesUrl && (
              <a
                href={currentPairing.seriesUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 bg-red-50 hover:bg-red-100 text-red-700 rounded-lg text-sm font-medium transition-colors"
              >
                <ExternalLink className="h-4 w-4" />
                Watch Full Series
              </a>
            )}
          </div>
        </div>

        <Button
          onClick={() => onPairingClick(currentPairing)}
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
          {pairings.map((pairing, index) => (
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
              aria-label={`Go to ${pairings[index].show} pairing${pairing.status !== 'available' ? ' (Coming Soon)' : ''}`}
            />
          ))}
        </div>

        {showSignedOutPrompt && (
          <SignedOut>
            <p className="text-xs text-primary font-medium text-center bg-primary/5 p-3 rounded">
              ✨ Preview lessons for free! Sign up to save your progress and unlock all pairings
            </p>
          </SignedOut>
        )}
      </div>
    </Card>
  );
};

export default ShowPairingCarousel;
