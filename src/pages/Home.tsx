import { Link } from 'react-router-dom';
import { Button } from '../components/ui/button';
import Footer from "../components/Footer";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '../components/ui/card';

function Home() {
  return (
    <div className="max-w-5xl mx-auto p-8 space-y-16">
      {/* Hero/Intro Section */}
      <section className="container-padding">
        <h1 className="text-4xl font-bold text-foreground">Progressive AI Academy</h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          A next-generation learning platform that leverages AI to deliver adaptive, personalized education from elementary through graduate levels. Built for modern learners, powered by cutting-edge technology, and designed for your success.
        </p>
        <div className="flex flex-wrap justify-center gap-4 pt-2">
          <Button asChild size="lg">
            <Link to="/onboarding">Get Started</Link>
          </Button>
          <Button asChild size="lg" variant="outline">
            <Link to="/sample-lesson">Try a Sample Lesson</Link>
          </Button>
          <Button asChild size="lg" variant="secondary">
            <Link to="/dashboard/lessons">Continue Learning</Link>
          </Button>
        </div>
      </section>

      {/* Features Overview */}
      <section className="grid md:grid-cols-2 gap-8">
        <Card className="overflow-hidden transition-all hover:shadow-md">
          <CardHeader>
            <CardTitle className="text-xl">Onboarding & Assessment</CardTitle>
            <CardDescription>Skill quiz to place learners at the right level</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-base text-foreground/90">
              Take a quick skill assessment to personalize your journey and start at the level that’s right for you.
            </p>
          </CardContent>
          <CardFooter className="bg-muted/20 pt-0">
            <Button asChild variant="outline" size="sm">
              <Link to="/onboarding">Start Assessment</Link>
            </Button>
          </CardFooter>
        </Card>
        <Card className="overflow-hidden transition-all hover:shadow-md">
          <CardHeader>
            <CardTitle className="text-xl">Adaptive Recommendations</CardTitle>
            <CardDescription>Claude-powered lesson suggestions</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-base text-foreground/90">
              Receive personalized lesson and project recommendations, powered by AI, to maximize your learning efficiency.
            </p>
          </CardContent>
          <CardFooter className="bg-muted/20 pt-0">
            <Button asChild variant="outline" size="sm">
              <Link to="/dashboard/lessons">See Recommendations</Link>
            </Button>
          </CardFooter>
        </Card>
        <Card className="overflow-hidden transition-all hover:shadow-md">
          <CardHeader>
            <CardTitle className="text-xl">Interactive Lesson Viewer</CardTitle>
            <CardDescription>Code sandboxes & live feedback</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-base text-foreground/90">
              Explore lessons with embedded code playgrounds and instant feedback to reinforce concepts as you learn.
            </p>
          </CardContent>
          <CardFooter className="bg-muted/20 pt-0">
            <Button asChild variant="outline" size="sm">
              <Link to="/sample-lesson">Try a Lesson</Link>
            </Button>
          </CardFooter>
        </Card>
        <Card className="overflow-hidden transition-all hover:shadow-md">
          <CardHeader>
            <CardTitle className="text-xl">Chat Tutor & Progress Dashboard</CardTitle>
            <CardDescription>AI Q&A and achievement tracking</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-base text-foreground/90">
              Get instant help from our AI-powered tutor and track your progress, achievements, and badges as you advance.
            </p>
          </CardContent>
          <CardFooter className="bg-muted/20 pt-0">
            <Button asChild variant="outline" size="sm">
              <Link to="/dashboard/lessons">Go to Dashboard</Link>
            </Button>
          </CardFooter>
        </Card>
      </section>

      {/* Value Proposition & How It Works */}
      <section className="space-y-8">
        <h2 className="text-2xl font-semibold text-foreground text-center">Why Progressive AI Academy?</h2>
        <div className="grid md:grid-cols-2 gap-8">
          <div className="space-y-4">
            <h3 className="text-xl font-semibold text-foreground">Personalized, Adaptive Learning</h3>
            <p className="text-base text-muted-foreground">
              Our platform adapts to your skill level, learning style, and pace. Every lesson and recommendation is tailored to help you succeed.
            </p>
            <h3 className="text-xl font-semibold text-foreground">Modern, Engaging Experience</h3>
            <p className="text-base text-muted-foreground">
              Enjoy interactive lessons, instant feedback, and a beautiful, distraction-free interface designed for real progress.
            </p>
          </div>
          <div className="space-y-4">
            <h3 className="text-xl font-semibold text-foreground">Powered by Leading AI</h3>
            <p className="text-base text-muted-foreground">
              Built on Anthropic Claude and Cloudflare’s edge network, our AI tutor and adaptive engine provide world-class support and recommendations.
            </p>
            <h3 className="text-xl font-semibold text-foreground">Track & Celebrate Your Growth</h3>
            <p className="text-base text-muted-foreground">
              Visualize your journey, unlock badges, and see your skills grow with our progress dashboard.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <Footer />
    </div>
  );
}

export default Home;
