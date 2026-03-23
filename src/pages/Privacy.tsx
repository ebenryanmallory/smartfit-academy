import { Link } from "react-router-dom";
import { motion } from "motion/react";
import { Shield, Mail, ExternalLink } from "lucide-react";

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: "easeOut" }
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.08
    }
  }
};

const sections = [
  {
    number: "1",
    title: "Information We Collect",
    content: (
      <>
        <h3 className="text-base font-semibold text-foreground mb-3">Information You Provide</h3>
        <p className="text-muted-foreground mb-3">When you create an account or use the platform, we collect:</p>
        <ul className="space-y-2 mb-6">
          {[
            { label: "Account information", desc: "your name and email address, provided when you register" },
            { label: "Authentication data", desc: "login credentials managed through Clerk (our authentication provider); if you sign in via Google, Apple, or another provider, we receive the profile information that provider shares with us" },
            { label: "Payment information", desc: "when you subscribe to a paid plan, your billing details are handled directly by Stripe through Clerk's pricing tables; we do not receive or store your card number, expiry date, or CVV" },
            { label: "Search queries and topic inputs", desc: "text you enter into the Modern Relevance search tool, the Dashboard topic explorer, or similar AI-powered features" },
          ].map((item) => (
            <li key={item.label} className="flex gap-2 text-sm text-muted-foreground">
              <span className="text-primary mt-0.5 flex-shrink-0">—</span>
              <span><span className="font-medium text-foreground">{item.label}</span> — {item.desc}</span>
            </li>
          ))}
        </ul>

        <h3 className="text-base font-semibold text-foreground mb-3">Information Collected Automatically</h3>
        <p className="text-muted-foreground mb-3">When you use the platform, we automatically collect:</p>
        <ul className="space-y-2">
          {[
            { label: "Usage data", desc: "pages and content you visit, features you use, time spent, clicks and navigation patterns" },
            { label: "Device and browser data", desc: "browser type and version, operating system, screen resolution, language settings" },
            { label: "IP address", desc: "used to determine your approximate location (country/region) and for security purposes" },
            { label: "Cookies and similar technologies", desc: "see Section 6 for details" },
          ].map((item) => (
            <li key={item.label} className="flex gap-2 text-sm text-muted-foreground">
              <span className="text-primary mt-0.5 flex-shrink-0">—</span>
              <span><span className="font-medium text-foreground">{item.label}</span> — {item.desc}</span>
            </li>
          ))}
        </ul>
      </>
    )
  },
  {
    number: "2",
    title: "How We Use Your Information",
    content: (
      <>
        <p className="text-muted-foreground mb-4">We use the information we collect to:</p>
        <ul className="space-y-2">
          {[
            "Create and manage your account",
            "Provide access to lessons, content tools, and platform features",
            "Process payments and manage your subscription",
            "Send transactional emails — account confirmation, password resets, and billing notifications",
            "Power AI-driven features such as the Modern Relevance search and Dashboard topic explorer (your query text is sent to an AI service to generate educational content connections)",
            "Analyse how the platform is used so we can improve it",
            "Respond to your support requests and communications",
            "Detect and prevent fraud, abuse, and security incidents",
            "Comply with our legal obligations",
          ].map((item) => (
            <li key={item} className="flex gap-2 text-sm text-muted-foreground">
              <span className="text-primary mt-0.5 flex-shrink-0">—</span>
              <span>{item}</span>
            </li>
          ))}
        </ul>
        <p className="text-sm text-muted-foreground mt-4 pt-4 border-t border-border/40 font-medium">
          We do not use your personal information to train or improve third-party AI models.
        </p>
      </>
    )
  },
  {
    number: "3",
    title: "Third-Party Services",
    content: (
      <>
        <p className="text-muted-foreground mb-6">We use the following third-party services to operate the platform. Each has its own privacy policy governing how it handles data.</p>
        <div className="overflow-x-auto -mx-1">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 pr-4 font-semibold text-foreground">Service</th>
                <th className="text-left py-3 pr-4 font-semibold text-foreground">Purpose</th>
                <th className="text-left py-3 font-semibold text-foreground">Privacy Policy</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {[
                { service: "Clerk", url: "clerk.com", purpose: "Account authentication, user management, and subscription billing", policy: "clerk.com/legal/privacy" },
                { service: "Stripe", url: "stripe.com", purpose: "Payment processing (via Clerk's pricing table integration)", policy: "stripe.com/privacy" },
                { service: "PostHog", url: "posthog.com", purpose: "Web analytics and product usage tracking", policy: "posthog.com/privacy" },
                { service: "OpenAI", url: "openai.com", purpose: "Processes search queries and topic inputs to generate educational content", policy: "openai.com/policies/privacy-policy" },
                { service: "Anthropic", url: "anthropic.com", purpose: "Processes search queries and topic inputs to generate educational content", policy: "anthropic.com/privacy" },
                { service: "Cloudflare", url: "cloudflare.com", purpose: "DNS, email routing, and security", policy: "cloudflare.com/privacypolicy" },
              ].map((row) => (
                <tr key={row.service} className="hover:bg-muted/30 transition-colors">
                  <td className="py-3 pr-4 font-medium text-foreground whitespace-nowrap">{row.service}</td>
                  <td className="py-3 pr-4 text-muted-foreground">{row.purpose}</td>
                  <td className="py-3">
                    <a
                      href={`https://${row.policy}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:text-primary/80 transition-colors inline-flex items-center gap-1 whitespace-nowrap"
                    >
                      {row.policy}
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-sm text-muted-foreground mt-4 pt-4 border-t border-border/40">
          We do not sell your personal information to any third party. We share data with the above services only to the extent necessary to operate the platform.
        </p>
      </>
    )
  },
  {
    number: "4",
    title: "Data Retention",
    content: (
      <div className="space-y-3 text-sm text-muted-foreground">
        <p>We retain your personal information for as long as your account is active or as needed to provide our services.</p>
        <p>If you delete your account, we will delete or anonymise your personal data within 30 days, except where we are required to retain it longer for legal or compliance reasons — for example, payment and billing records, which we retain for a minimum of 7 years as required by US tax law.</p>
        <p>Anonymous, aggregated usage data (which cannot be used to identify you) may be retained indefinitely for analytics purposes.</p>
      </div>
    )
  },
  {
    number: "5",
    title: "Your Privacy Rights",
    content: (
      <>
        <p className="text-sm text-muted-foreground mb-4">Regardless of where you are located, you have the right to:</p>
        <ul className="space-y-2 mb-4">
          {[
            { label: "Access", desc: "request a copy of the personal data we hold about you" },
            { label: "Correction", desc: "ask us to correct inaccurate or incomplete data" },
            { label: "Deletion", desc: "request that we delete your account and associated personal data (subject to legal retention requirements noted above)" },
            { label: "Opt out of marketing", desc: "unsubscribe from any marketing emails at any time using the unsubscribe link in the email, or by contacting us" },
          ].map((item) => (
            <li key={item.label} className="flex gap-2 text-sm text-muted-foreground">
              <span className="text-primary mt-0.5 flex-shrink-0">—</span>
              <span><span className="font-medium text-foreground">{item.label}</span> — {item.desc}</span>
            </li>
          ))}
        </ul>
        <p className="text-sm text-muted-foreground">
          To exercise any of these rights, email us at{" "}
          <a href="mailto:support@smartfit.academy" className="text-primary hover:text-primary/80 transition-colors">
            support@smartfit.academy
          </a>
          . We will respond within 30 days.
        </p>
      </>
    )
  },
  {
    number: "6",
    title: "Cookies",
    content: (
      <div className="space-y-4 text-sm text-muted-foreground">
        <p>We use cookies and similar browser storage technologies to operate the platform and understand how it is used.</p>
        <div className="rounded-lg bg-muted/40 border border-border/40 p-4 space-y-3">
          <div>
            <p className="font-medium text-foreground mb-1">Essential cookies</p>
            <p>Required for core functionality — keeping you logged in, maintaining session security, and remembering your preferences. These cannot be disabled without breaking the platform.</p>
          </div>
          <div className="border-t border-border/40 pt-3">
            <p className="font-medium text-foreground mb-1">Analytics cookies</p>
            <p>Via PostHog — help us understand how users navigate the site: which pages they visit, how long they spend, and where they drop off. This data is collected in aggregate and is used to improve the product.</p>
          </div>
        </div>
        <p>You can manage cookie settings in your browser. Note that disabling cookies may prevent you from logging in or using certain features.</p>
      </div>
    )
  },
  {
    number: "7",
    title: "Children's Privacy",
    content: (
      <div className="space-y-3 text-sm text-muted-foreground">
        <p>SmartFit Academy is intended for users aged 13 and older. We do not knowingly collect personal information from children under 13. If you believe a child under 13 has created an account, please contact us at <a href="mailto:support@smartfit.academy" className="text-primary hover:text-primary/80 transition-colors">support@smartfit.academy</a> and we will promptly delete the account and associated data.</p>
        <p>If you are a parent or educational institution using SmartFit Academy with students aged 13–17, please be aware that account creation requires a valid email address. We recommend parental or institutional oversight for users in this age range.</p>
      </div>
    )
  },
  {
    number: "8",
    title: "Security",
    content: (
      <>
        <p className="text-sm text-muted-foreground mb-4">We take reasonable technical and organisational measures to protect your personal information, including:</p>
        <ul className="space-y-2 mb-4">
          {[
            "HTTPS encryption on all pages",
            "Secure authentication managed by Clerk, including support for two-factor authentication",
            "Payment data handled exclusively by Stripe — we never see or store card details",
            "Access controls limiting who can access production systems and user data",
          ].map((item) => (
            <li key={item} className="flex gap-2 text-sm text-muted-foreground">
              <span className="text-primary mt-0.5 flex-shrink-0">—</span>
              <span>{item}</span>
            </li>
          ))}
        </ul>
        <p className="text-sm text-muted-foreground">No system can guarantee complete security. In the event of a data breach that affects your personal information, we will notify you as required by applicable law.</p>
      </>
    )
  },
  {
    number: "9",
    title: "Links to Other Websites",
    content: (
      <p className="text-sm text-muted-foreground">Our platform may contain links to third-party websites or services (for example, links to full books, films, or educational resources). This Privacy Policy does not cover those sites. We encourage you to review the privacy policies of any third-party sites you visit.</p>
    )
  },
  {
    number: "10",
    title: "Changes to This Policy",
    content: (
      <div className="space-y-3 text-sm text-muted-foreground">
        <p>We may update this Privacy Policy from time to time. If we make material changes, we will notify you by email or by posting a notice on the platform before the changes take effect. Your continued use of SmartFit Academy after that point constitutes your acceptance of the updated policy.</p>
        <p>The date at the top of this page indicates when the policy was last updated.</p>
      </div>
    )
  },
  {
    number: "11",
    title: "Contact",
    content: (
      <>
        <p className="text-sm text-muted-foreground mb-4">If you have questions or concerns about this Privacy Policy or how we handle your data, contact us at:</p>
        <div className="rounded-lg bg-muted/40 border border-border/40 p-4 space-y-1 text-sm">
          <p className="font-semibold text-foreground">SmartFit Academy</p>
          <p className="text-muted-foreground">Austin, Texas, USA</p>
          <p>
            <a href="mailto:support@smartfit.academy" className="text-primary hover:text-primary/80 transition-colors inline-flex items-center gap-1">
              <Mail className="h-3.5 w-3.5" />
              support@smartfit.academy
            </a>
          </p>
          <p>
            <a href="https://smartfit.academy/" target="_blank" rel="noopener noreferrer" className="text-primary hover:text-primary/80 transition-colors inline-flex items-center gap-1">
              <ExternalLink className="h-3.5 w-3.5" />
              smartfit.academy
            </a>
          </p>
        </div>
      </>
    )
  },
];

export default function Privacy() {
  return (
    <div className="flex flex-col min-h-screen bg-background">
      {/* Hero */}
      <section className="bg-gradient-to-br from-background via-palette-3 to-background relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-transparent to-accent/5" />
        <div className="container-section content-container-md relative z-10">
          <motion.div
            className="py-12 space-y-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="flex items-center gap-2 text-sm text-muted-foreground mb-2">
              <Shield className="h-4 w-4 text-primary" />
              <span>Legal</span>
            </div>
            <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold tracking-tight text-foreground leading-tight">
              Privacy{" "}
              <span className="relative text-primary font-light italic">
                Policy
                <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-primary to-accent rounded-full opacity-60 transform translate-y-2" />
              </span>
            </h1>
            <p className="text-base text-muted-foreground font-light">
              Last updated: March 2026
            </p>
          </motion.div>
        </div>
      </section>

      {/* Intro */}
      <section className="bg-white">
        <div className="container-section content-container-md">
          <motion.div
            className="prose prose-sm max-w-none space-y-3"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <p className="text-muted-foreground leading-relaxed">
              SmartFit Academy ("we," "our," or "us") operates the platform at{" "}
              <a href="https://smartfit.academy/" target="_blank" rel="noopener noreferrer" className="text-primary hover:text-primary/80 transition-colors">
                smartfit.academy
              </a>
              . This Privacy Policy explains what personal information we collect, how we use it, and what rights you have over it.
            </p>
            <p className="text-muted-foreground leading-relaxed">
              By using SmartFit Academy, you agree to the practices described in this policy. If you do not agree, please do not use the platform.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Sections */}
      <section className="bg-white pb-16">
        <div className="container-section content-container-md pt-0">
          <motion.div
            className="space-y-0"
            variants={staggerContainer}
            initial="initial"
            animate="animate"
          >
            {sections.map((section) => (
              <motion.div
                key={section.number}
                variants={fadeInUp}
                className="border-t border-border/50 py-8"
              >
                <div className="flex gap-4 md:gap-8">
                  <div className="flex-shrink-0 w-8 mt-0.5">
                    <span className="text-sm font-mono text-primary/60 font-semibold">{section.number}.</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <h2 className="text-lg font-semibold text-foreground mb-4">{section.title}</h2>
                    {section.content}
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>
    </div>
  );
}
