import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from "react-router-dom";
import { GraduationCap, Zap, Users, CheckCircle, BookOpen, Code, Brain, Rocket } from "lucide-react";
import Footer from "../components/Footer";

const features = [
  {
    icon: <GraduationCap className="h-8 w-8 text-primary mx-auto" />,
    title: "Personalized Lessons",
    desc: "AI adapts content to your learning pace and style."
  },
  {
    icon: <Zap className="h-8 w-8 text-accent mx-auto" />,
    title: "Instant Feedback",
    desc: "Get real-time feedback and suggested improvements."
  },
  {
    icon: <Users className="h-8 w-8 text-secondary mx-auto" />,
    title: "Community Support",
    desc: "Join a supportive community of learners and mentors."
  },
  {
    icon: <CheckCircle className="h-8 w-8 text-success mx-auto" />,
    title: "Progress Tracking",
    desc: "Visualize your progress and celebrate achievements."
  },
];

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-screen bg-background">
      {/* Hero Section */}
      <section className="container-section">
        <div className="content-container-md">
          <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto">
            Adaptive, personalized education for every learner. Our AI-powered platform delivers tailored lessons, instant feedback, and a seamless learning experience from foundational concepts to advanced research.
          </p>
          <div className="flex flex-wrap justify-center gap-4 mt-8">
            <Button className="btn-primary" asChild>
              <Link to="/dashboard">Get Started</Link>
            </Button>
            <Button variant="link" asChild>
              <Link to="/sample-lesson">Try Sample Lesson</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="container-section">
        <div className="content-container">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-10 text-foreground">Why Learn With Us?</h2>
          <div className="responsive-grid">
            {features.map((feature, idx) => (
              <div key={idx}>
                <Card className="feature-card">
                  <div className="mb-4">{feature.icon}</div>
                  <CardTitle className="text-xl font-bold mb-2">{feature.title}</CardTitle>
                  <CardDescription className="text-muted-foreground">{feature.desc}</CardDescription>
                </Card>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonial / Trust Section (Optional) */}
      <section className="container-section bg-background border-t border-border">
        <div className="content-container-sm text-center">
          <p className="text-lg md:text-xl italic text-muted-foreground mb-4">
            "Progressive AI Academy made learning Python feel effortless and fun. The instant feedback and personalized lessons kept me motivated every step of the way."
          </p>
          <div className="flex flex-col items-center gap-2">
            <span className="font-bold text-foreground">Alex Rivera</span>
            <span className="text-muted-foreground text-sm">Student, Data Science Enthusiast</span>
          </div>
        </div>
      </section>

      {/* Learning Path Overview */}
      <section className="container-section bg-secondary">
        <div className="content-container">
          <h2 className="text-3xl font-bold text-center mb-12">Your Learning Journey</h2>
          <div className="two-column-grid">
            {[
              { title: "Elementary", desc: "Build a strong foundation in logic, problem-solving, and basic programming. Fun, interactive, and beginner-friendly.", icon: <BookOpen className="h-6 w-6 text-primary mb-2" /> },
              { title: "High School", desc: "Master core programming skills, algorithms, and real-world applications. Prepare for advanced study and projects.", icon: <Code className="h-6 w-6 text-secondary mb-2" /> },
              { title: "Undergraduate", desc: "Tackle advanced concepts, hands-on projects, and collaborative learning. Bridge theory and practice with AI guidance.", icon: <Brain className="h-6 w-6 text-accent mb-2" /> },
              { title: "Graduate", desc: "Explore specialized topics, research opportunities, and cutting-edge technology with expert and AI mentorship.", icon: <Rocket className="h-6 w-6 text-success mb-2" /> },
            ].map((level) => (
              <Card key={level.title} className="card-base">
                <CardHeader className="flex flex-row items-center gap-2">
                  {level.icon}
                  <CardTitle>{level.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">{level.desc}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Features Overview */}
      <section className="container-section bg-accent">
        <div className="content-container-md">
          <h2 className="text-3xl font-bold text-center mb-12">What Makes Us Different?</h2>
          <div className="two-column-grid">
            <Card className="info-card">
              <CardHeader className="px-0 pt-0">
                <CardTitle className="text-xl font-semibold text-foreground">AI-Powered Personalization</CardTitle>
              </CardHeader>
              <CardContent className="px-0 pb-0">
                <p className="text-muted-foreground">Every lesson, quiz, and recommendation is tailored to your current skills and goals, ensuring an efficient, motivating journey.</p>
              </CardContent>
            </Card>
            <Card className="info-card">
              <CardHeader className="px-0 pt-0">
                <CardTitle className="text-xl font-semibold text-foreground">Interactive & Engaging</CardTitle>
              </CardHeader>
              <CardContent className="px-0 pb-0">
                <p className="text-muted-foreground">Code sandboxes, instant feedback, and real-time tutoring make learning hands-on and fun.</p>
              </CardContent>
            </Card>
            <Card className="info-card">
              <CardHeader className="px-0 pt-0">
                <CardTitle className="text-xl font-semibold text-foreground">Track Your Progress</CardTitle>
              </CardHeader>
              <CardContent className="px-0 pb-0">
                <p className="text-muted-foreground">Visual dashboards, badges, and achievements help you celebrate milestones and stay motivated.</p>
              </CardContent>
            </Card>
            <Card className="info-card">
              <CardHeader className="px-0 pt-0">
                <CardTitle className="text-xl font-semibold text-foreground">Accessible for All Levels</CardTitle>
              </CardHeader>
              <CardContent className="px-0 pb-0">
                <p className="text-muted-foreground">From absolute beginners to advanced researchers—our platform grows with you.</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Sign Up CTA */}
      <section className="container-section bg-palette-2">
        <div className="content-container text-center">
          <h2 className="text-3xl font-bold mb-6">Ready to Start Learning?</h2>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Join thousands of learners who are already advancing their education with AI-powered personalized learning.
          </p>
          <Button className="btn-primary" asChild>
          <Link to="/dashboard">Sign Up Now</Link>
          </Button>
        </div>
      </section>

      <div className="mt-auto">
        <Footer />
      </div>
    </div>
  );
}