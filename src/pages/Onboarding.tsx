import { useState } from "react";
import { Link } from "react-router-dom";
import { Card } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { useUser } from "@clerk/clerk-react";
import EducationLevelSelector from "../components/EducationLevelSelector";

const steps = [
  {
    title: "Welcome to SmartFit Academy!",
    content: "Let's get you started with the basics."
  },
  {
    title: "What to Expect",
    content: "Choose your topic of interest and create your own personalized educational journey. Whether it's programming, data science, or any other subject, you'll have the freedom to learn at your own pace and focus on what matters most to you."
  },
  {
    title: "Education Level",
    content: "Help us personalize your learning experience by selecting your current education level.",
    hasEducationSelect: true
  },
  {
    title: "Get Started",
    content: "Ready to begin your learning journey? Let's go!"
  }
];

export default function Onboarding() {
  const [step, setStep] = useState(0);
  const { isSignedIn } = useUser();
  const nextStep = () => setStep((s) => Math.min(s + 1, steps.length - 1));
  const prevStep = () => setStep((s) => Math.max(s - 1, 0));

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-50">
      <Card className="max-w-md w-full p-8">
        <h2 className="text-2xl font-bold mb-4">{steps[step].title}</h2>
        <p className="mb-6">
          {step === 0 && !isSignedIn 
            ? "To get the most out of your learning experience, we recommend creating an account. This allows you to save your educational journey, track your progress, and pick up exactly where you left off across all your devices."
            : steps[step].content
          }
        </p>
        {steps[step].hasEducationSelect && (
          <EducationLevelSelector variant="select" showLabel={true} />
        )}
        <div className="flex justify-between">
          {step === 0 && !isSignedIn ? (
            <>
              <Button variant="outline">Sign In</Button>
              <Button>Sign Up</Button>
            </>
          ) : (
            <>
              <Button onClick={prevStep} disabled={step === 0} variant="outline">Back</Button>
              {step < steps.length - 1 ? (
                <Button onClick={nextStep}>Next</Button>
              ) : (
                <Button asChild>
                  <Link to="/dashboard/lessons">Go to Lessons</Link>
                </Button>
              )}
            </>
          )}
        </div>
      </Card>
    </div>
  );
}
