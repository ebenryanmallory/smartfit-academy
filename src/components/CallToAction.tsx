import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

interface CallToActionProps {
  heading: string;
  body: string;
  primaryLabel: string;
  primaryHref: string;
  secondaryLabel?: string;
  secondaryHref?: string;
  primaryReplacement?: React.ReactNode;
  background?: string;
}

const CallToAction = ({
  heading,
  body,
  primaryLabel,
  primaryHref,
  secondaryLabel,
  secondaryHref,
  primaryReplacement,
  background = "bg-palette-2",
}: CallToActionProps) => {
  return (
    <section className={background}>
      <div className="container-section content-container text-center">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="space-y-6"
        >
          <h2 className="text-3xl font-bold text-foreground">{heading}</h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">{body}</p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            {primaryReplacement ?? (
              <Button className="btn-primary" size="lg" asChild>
                <Link to={primaryHref}>{primaryLabel}</Link>
              </Button>
            )}
            {secondaryLabel && secondaryHref && (
              <Button variant="outline" size="lg" asChild>
                <Link to={secondaryHref}>{secondaryLabel}</Link>
              </Button>
            )}
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default CallToAction;
