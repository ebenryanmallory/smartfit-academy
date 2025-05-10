import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Link } from "react-router-dom"

export default function LandingPage() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="py-20 px-4 md:px-6 lg:px-8 bg-gradient-to-b from-background to-muted">
        <div className="container mx-auto max-w-6xl">
          <div className="text-center space-y-6">
            <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
              Progressive AI Academy
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto">
              Personalized education powered by AI, from elementary through graduate levels.
              Learn at your own pace with adaptive lessons and real-time tutoring.
            </p>
            <div className="flex justify-center gap-4">
              <Button size="lg" asChild>
                <Link to="/dashboard">Get Started</Link>
              </Button>
              <Button size="lg" variant="outline" asChild>
                <Link to="/dashboard/lessons">View Sample Lesson</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Sample Lesson Preview */}
      <section className="py-16 px-4 md:px-6 lg:px-8">
        <div className="container mx-auto max-w-6xl">
          <h2 className="text-3xl font-bold text-center mb-12">Try a Sample Lesson</h2>
          <Card className="max-w-3xl mx-auto">
            <CardHeader>
              <CardTitle>Introduction to AI</CardTitle>
              <CardDescription>Learn the fundamentals of artificial intelligence</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="mb-4">
                Discover how AI is transforming our world and learn the basic concepts that power modern AI systems.
              </p>
              <Button className="w-full" asChild>
                <Link to="/dashboard/lessons">Start Learning</Link>
              </Button>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Learning Path Overview */}
      <section className="py-16 px-4 md:px-6 lg:px-8 bg-muted">
        <div className="container mx-auto max-w-6xl">
          <h2 className="text-3xl font-bold text-center mb-12">Your Learning Journey</h2>
          <div className="grid md:grid-cols-4 gap-6">
            {[
              { title: "Elementary", desc: "Foundational concepts and basic programming" },
              { title: "High School", desc: "Intermediate topics and practical applications" },
              { title: "Undergraduate", desc: "Advanced concepts and real-world projects" },
              { title: "Graduate", desc: "Specialized topics and research opportunities" },
            ].map((level) => (
              <Card key={level.title}>
                <CardHeader>
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

      {/* Sign Up CTA */}
      <section className="py-16 px-4 md:px-6 lg:px-8">
        <div className="container mx-auto max-w-6xl text-center">
          <h2 className="text-3xl font-bold mb-6">Ready to Start Learning?</h2>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Join thousands of learners who are already advancing their education with AI-powered personalized learning.
          </p>
          <Button size="lg" className="mx-auto" asChild>
            <Link to="/dashboard">Sign Up Now</Link>
          </Button>
        </div>
      </section>
    </div>
  )
} 