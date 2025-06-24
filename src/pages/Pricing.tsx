import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SignInButton } from "@/components/auth/SignInButton";
import { Link } from "react-router-dom";
import { Check, Zap, Users, Crown, MessageCircle, Brain, Rocket, Star, Shield, Clock } from "lucide-react";
import { motion } from "motion/react";
import { useUser, PricingTable } from "@clerk/clerk-react";

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

export default function Pricing() {
  const { isSignedIn } = useUser();

  const pricingTiers = [
    {
      name: "Free Forever",
      price: "$0",
      period: "forever",
      description: "Get started with AI-powered learning",
      icon: <Star className="h-6 w-6" />,
      features: [
        "Create custom lesson paths",
        "Explore any area of interest",
        "Progress tracking",
        "Use as a learning companion or teacher",
        "Mobile-friendly interface",
        "Basic achievement system"
      ],
      cta: isSignedIn ? "Get Started" : "Sign In to Start",
      ctaLink: "/dashboard",
      variant: "outline",
      popular: false
    },
    {
      name: "AI Upgrade",
      price: "$9.99",
      period: "per month",
      description: "Unlock the power of state-of-the-art AI models",
      icon: <Brain className="h-6 w-6" />,
      features: [
        "Everything in Free Forever",
        "Access to OpenAI GPT-4 & Claude Sonnet",
        "Advanced reasoning capabilities",
        "Superior content generation",
        "Priority AI response times",
        "Enhanced lesson personalization",
        "Advanced progress analytics",
        "Premium achievement badges",
        "Early access to new features"
      ],
      highlights: [
        "Best-in-class AI models",
        "Always updated to latest models",
        "Optimized for academic content"
      ],
      cta: "Upgrade Now",
      ctaLink: "#",
      variant: "primary",
      popular: true
    },
    {
      name: "Enterprise",
      price: "Custom",
      period: "one-time",
      description: "Integrate our AI-powered engine into your business",
      icon: <Rocket className="h-6 w-6" />,
      features: [
        "Highly customizable functionality",
        "Include your documents for AI reference",
        "One time purchase pricing"
      ],
      highlights: [
        "Fully customizable platform",
        "Enterprise-grade security",
        "Scalable infrastructure"
      ],
      cta: "Contact Sales",
      ctaLink: "#contact",
      variant: "secondary",
      popular: false
    }
  ];

  const handleContactSales = () => {
    // This would typically open a contact form or redirect to a contact page
    window.location.href = "mailto:sales@smartfitacademy.com?subject=Enterprise Inquiry";
  };

  return (
    <div className="flex flex-col min-h-screen bg-background">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-background via-palette-3 to-background relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-transparent to-accent/5"></div>
        <div className="container-section content-container relative z-10">
          <motion.div 
            className="text-center space-y-6 py-12"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold tracking-tight text-foreground leading-tight">
              Choose Your <span className="relative text-primary font-light italic">
                Learning Journey
                <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-primary to-accent rounded-full opacity-60 transform translate-y-2"></div>
              </span>
            </h1>
            <p className="text-base md:text-lg text-muted-foreground leading-relaxed font-light max-w-2xl mx-auto">
              From free access to enterprise solutions, we have the perfect plan to accelerate your learning with AI-powered education.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Pricing Cards */}
      <section className="bg-white">
        <div className="container-section content-container">
          <motion.div 
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
            variants={staggerContainer}
            initial="initial"
            animate="animate"
          >
          {pricingTiers.map((tier, index) => (
            <motion.div
              key={tier.name}
              variants={fadeInUp}
              className={`relative ${tier.popular ? 'lg:scale-105' : ''}`}
            >
              {tier.popular && (
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 z-10">
                  <span className="bg-gradient-to-r from-primary to-accent text-white px-4 py-1 rounded-full text-sm font-medium shadow-lg">
                    Most Popular
                  </span>
                </div>
              )}
              
              <Card className={`${tier.popular ? 'border-primary shadow-xl' : 'border-border'} transition-all duration-300 hover:shadow-lg hover:-translate-y-1 overflow-hidden p-0 min-h-[450px] flex flex-col`}>
                <CardHeader className={`${tier.name === "AI Upgrade" ? "text-center" : "text-left"} mt-4`}>
                  <div className={`w-12 h-12 mx-auto rounded-full flex items-center justify-center mb-4 z-5 ${
                    tier.popular ? 'bg-primary text-white' : 'bg-white shadow-sm border border-gray-100 text-accent'
                  }`}>
                    {tier.icon}
                  </div>
                  {!tier.popular && (
                    <>
                    <CardTitle className="text-xl font-medium text-left">{tier.name}</CardTitle>
                    <CardDescription className="mt-2 text-left">{tier.description}</CardDescription>
                    <div className="flex items-baseline justify-start gap-1 mt-3">
                      <span className="text-2xl font-bold text-foreground">{tier.price}</span>
                      <span className="text-muted-foreground text-sm">/{tier.period}</span>
                    </div>
                    </>
                  )}
                </CardHeader>

                {tier.name === "AI Upgrade" ? (
                  <>
                    {/* Full-width PricingTable at bottom */}
                    <div className="-mt-16">
                       <PricingTable 
                         newSubscriptionRedirectUrl="/dashboard"
                         appearance={{
                           elements: {
                             // Main container
                             pricingTable: "w-full",
                             
                             // Individual plan cards - make them blend seamlessly
                             card: "border-0 shadow-none rounded-none bg-transparent",
                             cardHeader: "hidden", // Hide the header since we show plan info above
                             cardBody: "p-0 pt-4",
                             cardFooter: "border-t-0 bg-transparent p-4",
                             
                             // Plan content
                             pricingSlot: "bg-transparent border-0",
                             pricingSlotHeader: "hidden",
                             pricingSlotBody: "p-0",
                             pricingSlotFooter: "border-t border-border/20 mt-4 pt-4",
                             
                             // Features list - hide since we show above
                             pricingSlotFeature: "hidden",
                             pricingSlotFeatureList: "hidden",
                             
                             // Price display - minimal
                             pricingSlotPrice: "text-sm text-muted-foreground mb-2",
                             
                             // CTA Button - make it prominent
                             button: "w-full bg-gradient-to-r from-primary to-accent text-white font-semibold py-3 px-6 rounded-md shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-[1.02]"
                           }
                         }}
                         checkoutProps={{
                           appearance: {
                             elements: {
                               card: "rounded-xl shadow-2xl border border-border",
                               button: "bg-primary text-primary-foreground hover:bg-accent transition-colors"
                             }
                           }
                         }}
                       />
                     </div>
                  </>
                ) : (
                  <CardContent className="space-y-4 px-4 py-4">
                    {/* Features */}
                    <div className="border-t border-b border-stone-200 py-6 px-2">
                      <ul className="space-y-1.5">
                        {tier.features.map((feature, idx) => (
                          <li key={idx} className="flex items-start gap-2 text-sm">
                            <Check className="h-4 w-4 text-success flex-shrink-0 mt-0.5" />
                            <span className="text-left">{feature}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* CTA Button */}
                    <div className="pt-3">
                      {tier.name === "Free Forever" && !isSignedIn ? (
                        <SignInButton />
                      ) : tier.name === "Enterprise" ? (
                        <Button 
                          variant={tier.variant as any}
                          size="lg" 
                          className="w-full font-medium"
                          onClick={handleContactSales}
                        >
                          <MessageCircle className="h-4 w-4 mr-2" />
                          {tier.cta}
                        </Button>
                      ) : tier.name === "Free Forever" ? (
                        <Button 
                          variant={tier.variant as any}
                          size="lg" 
                          className="w-full font-medium"
                          asChild
                        >
                          <Link to={tier.ctaLink}>
                            {tier.cta}
                          </Link>
                        </Button>
                      ) : null}
                    </div>
                  </CardContent>
                )}
              </Card>
            </motion.div>
          ))}
          </motion.div>
        </div>
      </section>



      {/* FAQ Section */}
      <section className="bg-gradient-to-r from-palette-4/10 via-background to-palette-5/10">
        <div className="container-section content-container-md">
          <motion.div 
            className="text-center space-y-8"
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl md:text-3xl font-semibold text-foreground">
              Frequently Asked Questions
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-left">
              <div className="space-y-4">
                <h3 className="font-semibold text-primary">What makes your AI different?</h3>
                <p className="text-muted-foreground text-sm">
                  We use the latest and most capable AI models from OpenAI and Anthropic, specifically optimized for educational content creation and personalized learning.
                </p>
              </div>
              
              <div className="space-y-4">
                <h3 className="font-semibold text-primary">Can I upgrade or downgrade anytime?</h3>
                <p className="text-muted-foreground text-sm">
                  Yes! You can change your plan at any time. Upgrades take effect immediately, and downgrades take effect at the next billing cycle.
                </p>
              </div>
              
              <div className="space-y-4">
                <h3 className="font-semibold text-primary">What's included in Enterprise?</h3>
                <p className="text-muted-foreground text-sm">
                  Enterprise includes a fully white-labeled platform, custom AI training, dedicated support, and integration capabilities for your existing systems.
                </p>
              </div>
              
              <div className="space-y-4">
                <h3 className="font-semibold text-primary">Is my data secure?</h3>
                <p className="text-muted-foreground text-sm">
                  Absolutely. We use enterprise-grade security, encrypt all data, and never share your personal information or learning progress with third parties.
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-gradient-to-br from-primary/5 to-accent/5">
        <div className="container-section content-container-sm text-center">
          <motion.div 
            className="space-y-6"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl md:text-3xl font-semibold text-foreground">
              Ready to Transform Your Learning?
            </h2>
            <p className="text-muted-foreground">
              Join thousands of learners who are already experiencing the future of education with AI-powered personalized learning.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              {!isSignedIn ? (
                <SignInButton />
              ) : (
                <Button size="lg" className="btn-primary" asChild>
                  <Link to="/dashboard">Start Learning Now</Link>
                </Button>
              )}
              <Button variant="outline" size="lg" asChild>
                <Link to="/lessons/c-intro-ai">Try a Free Lesson</Link>
              </Button>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
} 