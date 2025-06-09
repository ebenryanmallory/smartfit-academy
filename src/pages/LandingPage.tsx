import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from "react-router-dom";
import { GraduationCap, Zap, Users, CheckCircle, BookOpen, Code, Brain, Rocket, Target, Clock, Award, Heart, TrendingUp } from "lucide-react";
import Footer from "../components/Footer";
import { motion } from "motion/react";

const features = [
  {
    icon: <Target className="h-8 w-8 text-primary mx-auto" />,
    title: "Learn at Your Own Pace",
    desc: "No pressure, no rush. Our AI adapts to how fast or slow you want to go, making sure you truly understand each concept before moving forward."
  },
  {
    icon: <Heart className="h-8 w-8 text-accent mx-auto" />,
    title: "Build Real Confidence",
    desc: "Get instant, encouraging feedback that helps you learn from mistakes without judgment. Every small win is celebrated."
  },
  {
    icon: <Users className="h-8 w-8 text-secondary mx-auto" />,
    title: "AI Learning Companion",
    desc: "Get instant help and guidance from our AI tutor whenever you're stuck. Ask questions anytime and get personalized explanations."
  },
  {
    icon: <TrendingUp className="h-8 w-8 text-success mx-auto" />,
    title: "See Your Growth",
    desc: "Track your progress with visual milestones and achievements that show how far you've come, not just how far you have to go."
  },
];

// Animation variants for consistent animations
const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: "easeOut" }
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
};

const fadeInScale = {
  initial: { opacity: 0, scale: 0.8 },
  animate: { opacity: 1, scale: 1 },
  transition: { duration: 0.5, ease: "easeOut" }
};

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-screen bg-background">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-background via-palette-3 to-background relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-transparent to-accent/5"></div>
        <div className="container-section content-container relative z-10">
                        <motion.div 
                className="space-y-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold tracking-tight text-foreground leading-tight">
                  Learn Without <span className="relative text-primary font-light italic">
                    Limits
                    <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-primary to-accent rounded-full opacity-60 transform translate-y-2"></div>
                  </span>
                </h1>
              </motion.div>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start py-12 md:py-20">
            {/* Main Content - One third width */}
            <motion.div 
              className="space-y-8"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, ease: "easeOut" }}
            >
              
              <motion.p 
                className="text-base md:text-lg text-muted-foreground leading-relaxed font-light"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
              >
                Whether you're just starting out or pushing boundaries, Smartfit Academy meets you exactly where you are. 
                <span className="block mt-3 text-foreground font-medium">AI-powered learning that actually gets you.</span>
              </motion.p>
              
              <motion.div 
                className="flex flex-col gap-3 pt-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.6 }}
              >
                <Button size="lg" className="btn-primary text-base px-6 py-4 shadow-lg hover:shadow-xl transition-all duration-300" asChild>
                  <Link to="/dashboard">Start Your Journey</Link>
                </Button>
                <Button variant="outline" size="lg" className="text-base px-6 py-4 border-2 border-primary/20 hover:border-primary/40 transition-all duration-300" asChild>
                  <Link to="/sample-lesson">Try a Free Lesson</Link>
                </Button>
              </motion.div>
            </motion.div>

            {/* Feature 1 - One third width */}
            <motion.div 
              className="space-y-4"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.7 }}
            >
              <div className="p-6 rounded-lg bg-card/30 backdrop-blur-sm border border-border/20 h-full">
                <Clock className="h-8 w-8 text-primary mb-4" />
                <h3 className="font-semibold text-foreground mb-2">Learn Anytime</h3>
                <p className="text-sm text-muted-foreground">24/7 access to all content with flexible scheduling that fits your lifestyle.</p>
              </div>
            </motion.div>

            {/* Feature 2 - One third width */}
            <motion.div 
              className="space-y-4"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.8 }}
            >
              <div className="p-6 rounded-lg bg-card/30 backdrop-blur-sm border border-border/20 h-full">
                <CheckCircle className="h-8 w-8 text-success mb-4" />
                <h3 className="font-semibold text-foreground mb-2">No Prerequisites</h3>
                <p className="text-sm text-muted-foreground">Start from any level and progress at your own pace with personalized guidance.</p>
              </div>
            </motion.div>
          </div>

          {/* Second row - 3 equal columns */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pb-12">
            <motion.div 
              className="text-center p-4 rounded-lg bg-card/20 backdrop-blur-sm"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.9 }}
            >
              <div className="flex items-center justify-center gap-2 text-primary mb-2">
                <Award className="h-6 w-6" />
                <span className="font-semibold text-lg">Earn Rewards</span>
              </div>
              <p className="text-sm text-muted-foreground">Unlock badges and achievements as you progress</p>
            </motion.div>
            
            <motion.div 
              className="text-center p-4 rounded-lg bg-card/20 backdrop-blur-sm"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 1.0 }}
            >
              <div className="flex items-center justify-center gap-2 text-accent mb-2">
                <Brain className="h-6 w-6" />
                <span className="font-semibold text-lg">AI-Powered</span>
              </div>
              <p className="text-sm text-muted-foreground">Personalized learning paths that adapt to you</p>
            </motion.div>
            
            <motion.div 
              className="text-center p-4 rounded-lg bg-card/20 backdrop-blur-sm"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 1.1 }}
            >
              <div className="flex items-center justify-center gap-2 text-success mb-2">
                <TrendingUp className="h-6 w-6" />
                <span className="font-semibold text-lg">Progress Tracking</span>
              </div>
              <p className="text-sm text-muted-foreground">Visual milestones and achievement system</p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="bg-gradient-to-b from-background to-secondary/10">
        <div className="container-section content-container">
          <motion.div 
            className="text-center mb-16 space-y-4"
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-foreground tracking-tight">
              Why Students Choose Us
            </h2>
            <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto font-light">
              We understand the real challenges you face when learning something new
            </p>
            <div className="w-16 h-0.5 bg-gradient-to-r from-primary to-accent mx-auto rounded-full opacity-50"></div>
          </motion.div>
          
          <motion.div 
            className="responsive-grid"
            variants={staggerContainer}
            initial="initial"
            whileInView="animate"
            viewport={{ once: true, margin: "-50px" }}
          >
            {features.map((feature, idx) => (
              <motion.div key={idx} className="group" variants={fadeInUp}>
                <Card className="feature-card h-full border-0 shadow-sm hover:shadow-lg transition-all duration-500 bg-card/50 backdrop-blur-sm hover:bg-card/80 hover:-translate-y-1">
                  <div className="mb-6 p-3 rounded-full bg-gradient-to-br from-primary/10 to-accent/10 w-fit mx-auto group-hover:scale-110 transition-transform duration-300">
                    {feature.icon}
                  </div>
                  <CardTitle className="text-xl font-semibold mb-4 text-center text-foreground group-hover:text-primary transition-colors duration-300">
                    {feature.title}
                  </CardTitle>
                  <CardDescription className="text-muted-foreground text-center leading-relaxed">
                    {feature.desc}
                  </CardDescription>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Original Hero Content - Now as Value Proposition */}
      <section className="bg-gradient-to-r from-palette-4/10 via-background to-palette-5/10">
        <div className="container-section content-container-md">
          <motion.div 
            className="text-center space-y-6 py-8"
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl md:text-3xl font-semibold text-foreground mb-4">
              Adaptive Learning That Actually Works
            </h2>
            <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Our AI-powered platform delivers tailored lessons, instant feedback, and a seamless learning experience from foundational concepts to advanced research.
            </p>
            <motion.div 
              className="flex flex-wrap justify-center gap-4 mt-8"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <Button className="btn-primary shadow-md hover:shadow-lg transition-all duration-300" asChild>
                <Link to="/dashboard">Get Started</Link>
              </Button>
              <Button variant="link" className="text-primary hover:text-primary/80 transition-colors duration-300" asChild>
                <Link to="/sample-lesson">Try Sample Lesson</Link>
              </Button>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Learning Path Overview */}
      <section className="bg-secondary">
        <div className="container-section content-container">
          <motion.h2 
            className="text-3xl font-bold text-center mb-12"
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
          >
            Your Learning Journey
          </motion.h2>
          <motion.div 
            className="two-column-grid"
            variants={staggerContainer}
            initial="initial"
            whileInView="animate"
            viewport={{ once: true, margin: "-50px" }}
          >
            {[
              { title: "Elementary", desc: "Build a strong foundation in logic, problem-solving, and basic programming. Fun, interactive, and beginner-friendly.", icon: <BookOpen className="h-6 w-6 text-primary mb-2" /> },
              { title: "High School", desc: "Master core programming skills, algorithms, and real-world applications. Prepare for advanced study and projects.", icon: <Code className="h-6 w-6 text-secondary mb-2" /> },
              { title: "Undergraduate", desc: "Tackle advanced concepts, hands-on projects, and collaborative learning. Bridge theory and practice with AI guidance.", icon: <Brain className="h-6 w-6 text-accent mb-2" /> },
              { title: "Graduate", desc: "Explore specialized topics, research opportunities, and cutting-edge technology with expert and AI mentorship.", icon: <Rocket className="h-6 w-6 text-success mb-2" /> },
            ].map((level) => (
              <motion.div key={level.title} variants={fadeInScale}>
                <Card className="card-base">
                  <CardHeader className="flex flex-row items-center gap-2">
                    {level.icon}
                    <CardTitle>{level.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">{level.desc}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features Overview */}
      <section className="bg-accent">
        <div className="container-section content-container-md">
          <motion.h2 
            className="text-3xl font-bold text-center mb-12"
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
          >
            What Makes Us Different?
          </motion.h2>
          <motion.div 
            className="two-column-grid"
            variants={staggerContainer}
            initial="initial"
            whileInView="animate"
            viewport={{ once: true, margin: "-50px" }}
          >
            <motion.div variants={fadeInUp}>
              <Card className="info-card">
                <CardHeader className="px-0 pt-0">
                  <CardTitle className="text-xl font-semibold text-foreground">AI-Powered Personalization</CardTitle>
                </CardHeader>
                <CardContent className="px-0 pb-0">
                  <p className="text-muted-foreground">Every lesson, quiz, and recommendation is tailored to your current skills and goals, ensuring an efficient, motivating journey.</p>
                </CardContent>
              </Card>
            </motion.div>
            <motion.div variants={fadeInUp}>
              <Card className="info-card">
                <CardHeader className="px-0 pt-0">
                  <CardTitle className="text-xl font-semibold text-foreground">Interactive & Engaging</CardTitle>
                </CardHeader>
                <CardContent className="px-0 pb-0">
                  <p className="text-muted-foreground">Code sandboxes, instant feedback, and real-time tutoring make learning hands-on and fun.</p>
                </CardContent>
              </Card>
            </motion.div>
            <motion.div variants={fadeInUp}>
              <Card className="info-card">
                <CardHeader className="px-0 pt-0">
                  <CardTitle className="text-xl font-semibold text-foreground">Track Your Progress</CardTitle>
                </CardHeader>
                <CardContent className="px-0 pb-0">
                  <p className="text-muted-foreground">Visual dashboards, badges, and achievements help you celebrate milestones and stay motivated.</p>
                </CardContent>
              </Card>
            </motion.div>
            <motion.div variants={fadeInUp}>
              <Card className="info-card">
                <CardHeader className="px-0 pt-0">
                  <CardTitle className="text-xl font-semibold text-foreground">Accessible for All Levels</CardTitle>
                </CardHeader>
                <CardContent className="px-0 pb-0">
                  <p className="text-muted-foreground">From absolute beginners to advanced researchers—our platform grows with you.</p>
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Sign Up CTA */}
      <section className="bg-palette-2">
        <div className="container-section content-container text-center">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-3xl font-bold mb-6">Ready to Start Learning?</h2>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Join thousands of learners who are already advancing their education with AI-powered personalized learning.
            </p>
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <Button className="btn-primary" asChild>
                <Link to="/dashboard">Sign Up Now</Link>
              </Button>
            </motion.div>
          </motion.div>
        </div>
      </section>

      <div className="mt-auto">
        <Footer />
      </div>
    </div>
  );
}