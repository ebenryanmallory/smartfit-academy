import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { SignInButton } from "@clerk/clerk-react";
import { Bookmark, ArrowRight } from "lucide-react";

interface SaveProgressPromptProps {
  title?: string;
  description?: string;
  className?: string;
  buttonText?: string;
  /**
   * Button style variant, e.g. 'default', 'secondary', 'link'.
   * Defaults to 'default'.
   */
  variant?: "default" | "secondary" | "link" | "destructive" | "ghost";
}

export function SaveProgressPrompt({ 
  title = "Save Your Progress", 
  description = "Sign in to save your progress and continue learning later.",
  className = "",
  buttonText = "Sign in to save progress",
  variant = "ghost"
}: SaveProgressPromptProps) {
  return (
    <Card className={`p-4 border-dashed ${className}`}>
      <div className="flex items-start gap-4">
        <Bookmark className="h-5 w-5 text-primary mt-1" />
        <div className="flex-1">
          <h3 className="font-medium">{title}</h3>
          <p className="text-sm text-muted-foreground mt-1">{description}</p>
          <SignInButton mode="modal">
            <Button className="p-4" variant={variant}>
              {buttonText}
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </SignInButton>
        </div>
      </div>
    </Card>
  );
} 