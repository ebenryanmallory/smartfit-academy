import { useState } from "react";
import { Card } from "../components/ui/card";
import { Button } from "../components/ui/button";

const steps = [
  {
    title: "Welcome to Progressive AI Academy!",
    content: "Let's get you started with the basics."
  },
  {
    title: "What to Expect",
    content: "You'll learn Python programming through interactive lessons and hands-on practice."
  },
  {
    title: "Get Started",
    content: "Ready to begin your learning journey? Let's go!"
  }
];

export default function Onboarding() {
  const [step, setStep] = useState(0);
  const nextStep = () => setStep((s) => Math.min(s + 1, steps.length - 1));
  const prevStep = () => setStep((s) => Math.max(s - 1, 0));

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-50">
      <Card className="max-w-md w-full p-8">
        <h2 className="text-2xl font-bold mb-4">{steps[step].title}</h2>
        <p className="mb-6">{steps[step].content}</p>
        <div className="flex justify-between">
          <Button onClick={prevStep} disabled={step === 0} variant="outline">Back</Button>
          {step < steps.length - 1 ? (
            <Button onClick={nextStep}>Next</Button>
          ) : (
            <Button asChild>
              <a href="/Lessons">Go to Lessons</a>
            </Button>
          )}
        </div>
      </Card>
    </div>
  );
}
