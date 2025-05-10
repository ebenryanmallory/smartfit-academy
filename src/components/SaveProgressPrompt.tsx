import React from 'react';
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { Bookmark, ArrowRight } from "lucide-react";

interface SaveProgressPromptProps {
  title?: string;
  description?: string;
  className?: string;
  buttonText?: string;
}

export function SaveProgressPrompt({ 
  title = "Save Your Progress", 
  description = "Sign in to save your progress and continue learning later.",
  className = "",
  buttonText = "Sign in to save progress"
}: SaveProgressPromptProps) {
  return (
    <Card className={`p-4 bg-muted/50 border-dashed ${className}`}>
      <div className="flex items-start gap-4">
        <Bookmark className="h-5 w-5 text-primary mt-1" />
        <div className="flex-1">
          <h3 className="font-medium">{title}</h3>
          <p className="text-sm text-muted-foreground mt-1">{description}</p>
          <Button asChild className="mt-3">
            <Link to="/dashboard">
              {buttonText}
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>
    </Card>
  );
} 