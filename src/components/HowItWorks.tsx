import React from "react";
import { motion } from "framer-motion";
import { Search, GitBranch, BookOpen } from "lucide-react";

interface HowItWorksProps {
  heading?: string;
}

const steps = [
  {
    number: "1",
    icon: Search,
    title: "You Share What's Trending",
    description:
      "Input any current topic, trend, or controversy you've seen online or in the news.",
    color: "primary",
  },
  {
    number: "2",
    icon: GitBranch,
    title: "AI Finds Historical Parallels",
    description:
      "Our engine analyzes the topic and identifies relevant historical events, thinkers, and texts.",
    color: "accent",
  },
  {
    number: "3",
    icon: BookOpen,
    title: "You Get Custom Lessons",
    description:
      "Receive personalized learning paths that connect modern issues to timeless wisdom.",
    color: "secondary",
  },
];

const colorMap: Record<string, { bg: string; text: string; border: string }> = {
  primary: {
    bg: "bg-primary/10",
    text: "text-primary",
    border: "border-primary/20",
  },
  accent: {
    bg: "bg-accent/10",
    text: "text-accent",
    border: "border-accent/20",
  },
  secondary: {
    bg: "bg-secondary/10",
    text: "text-secondary",
    border: "border-secondary/20",
  },
};

const stagger = {
  animate: { transition: { staggerChildren: 0.15 } },
};

const fadeInUp = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.55, ease: "easeOut" },
};

const HowItWorks: React.FC<HowItWorksProps> = ({
  heading = "How The Relevance Engine Works",
}) => {
  return (
    <section className="container-section bg-gradient-to-r from-accent/10 to-primary/10">
      <div className="content-container">
        <motion.h2
          className="text-3xl md:text-4xl font-bold text-center mb-10 text-foreground"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.55 }}
        >
          {heading}
        </motion.h2>

        <motion.div
          className="grid md:grid-cols-3 gap-8"
          variants={stagger}
          initial="initial"
          whileInView="animate"
          viewport={{ once: true, margin: "-60px" }}
        >
          {steps.map((step) => {
            const IconComponent = step.icon;
            const colors = colorMap[step.color];
            return (
              <motion.div
                key={step.number}
                className="text-center"
                variants={fadeInUp}
              >
                <div
                  className={`w-16 h-16 ${colors.bg} border-2 ${colors.border} rounded-full flex items-center justify-center mx-auto mb-4`}
                >
                  <IconComponent className={`h-7 w-7 ${colors.text}`} />
                </div>
                <h3 className="text-xl font-semibold mb-3">{step.title}</h3>
                <p className="text-muted-foreground">{step.description}</p>
              </motion.div>
            );
          })}
        </motion.div>
      </div>
    </section>
  );
};

export default HowItWorks;
