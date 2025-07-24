import { Card } from "@/components/ui/card";
import { SignInButton } from "@/components/auth/SignInButton";
import { Bookmark } from "lucide-react";

interface SaveProgressPromptProps {
  title?: string;
  description?: string;
  className?: string;
}

export function SaveProgressPrompt({ 
  title = "Save Your Progress", 
  description = "Sign in to save your progress and continue learning later.",
  className = ""
}: SaveProgressPromptProps) {
  return (
    <Card className={`p-4 border-dashed ${className}`}>
      <div className="flex items-start gap-4">
        <Bookmark className="h-5 w-5 text-primary mt-1" />
        <div className="flex-1">
          <h3 className="font-medium">{title}</h3>
          <p className="text-sm text-muted-foreground mt-1">{description}</p>
          <div className="mt-3">
            <SignInButton />
          </div>
        </div>
      </div>
    </Card>
  );
} 