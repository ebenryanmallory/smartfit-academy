import { Button } from './ui/button';

interface CreateYourJourneyProps {
  onExpandAssistant: () => void;
}

function CreateYourJourney({ onExpandAssistant }: CreateYourJourneyProps) {
  return (
    <section className="container-section bg-gradient-to-r from-primary/10 to-accent/10">
      <div className="content-container text-center">
        <h2 className="text-4xl md:text-5xl font-bold mb-6 text-foreground">
          Create Your Educational Journey
        </h2>
        <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto">
          Start exploring topics that interest you. Our AI assistant will help you discover new areas of learning and build a personalized curriculum just for you.
        </p>
        <Button 
          size="lg" 
          className="text-lg px-8 py-4 h-auto"
          onClick={onExpandAssistant}
        >
          Start Exploring Topics
        </Button>
      </div>
    </section>
  );
}

export default CreateYourJourney;