import { useState, useEffect } from "react";
import { useUser, useAuth } from "@clerk/clerk-react";
import { toast } from "sonner";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "./ui/dropdown-menu";
import { Button } from "./ui/button";
import { ChevronDown, GraduationCap } from "lucide-react";

// Define the mapping between backend values and display values
export type EducationLevel = 'elementary' | 'highschool' | 'undergrad' | 'grad';
export type AudienceLevel = 'elementary' | 'high-school' | 'undergraduate' | 'graduate';

const educationLevels: { value: EducationLevel; label: string; icon: string; audienceValue: AudienceLevel }[] = [
  { value: 'elementary', label: 'Elementary School', icon: '🎓', audienceValue: 'elementary' },
  { value: 'highschool', label: 'High School', icon: '📚', audienceValue: 'high-school' },
  { value: 'undergrad', label: 'Undergraduate', icon: '🎯', audienceValue: 'undergraduate' },
  { value: 'grad', label: 'Graduate', icon: '🔬', audienceValue: 'graduate' }
];

// Mapping functions
const educationToAudience = (education: EducationLevel): AudienceLevel => {
  const mapping = educationLevels.find(level => level.value === education);
  return mapping?.audienceValue || 'undergraduate';
};

const audienceToEducation = (audience: AudienceLevel): EducationLevel => {
  const mapping = educationLevels.find(level => level.audienceValue === audience);
  return mapping?.value || 'undergrad';
};

interface EducationLevelSelectorProps {
  value?: AudienceLevel;
  onChange?: (value: AudienceLevel) => void;
  variant?: 'dropdown' | 'select';
  showLabel?: boolean;
  className?: string;
  disabled?: boolean;
}

export default function EducationLevelSelector({
  value,
  onChange,
  variant = 'dropdown',
  showLabel = true,
  className = '',
  disabled = false
}: EducationLevelSelectorProps) {
  const { isSignedIn } = useUser();
  const { getToken } = useAuth();
  const [currentLevel, setCurrentLevel] = useState<AudienceLevel>(value || 'undergraduate');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Fetch user's saved education level on mount
  useEffect(() => {
    const fetchUserEducationLevel = async () => {
      if (!isSignedIn) return;
      
      setIsLoading(true);
      try {
        const token = await getToken();
        const response = await fetch('/api/d1/user', {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        });

        if (response.ok) {
          const data = await response.json();
          if (data.user?.education_level) {
            const audienceLevel = educationToAudience(data.user.education_level as EducationLevel);
            setCurrentLevel(audienceLevel);
            onChange?.(audienceLevel);
          }
        }
      } catch (error) {
        console.error('Error fetching user education level:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchUserEducationLevel();
  }, [isSignedIn, getToken, onChange]);

  // Update local state when value prop changes
  useEffect(() => {
    if (value !== undefined) {
      setCurrentLevel(value);
    }
  }, [value]);

  const handleEducationLevelChange = async (newAudienceLevel: AudienceLevel) => {
    setCurrentLevel(newAudienceLevel);
    onChange?.(newAudienceLevel);
    
    if (isSignedIn) {
      setIsSubmitting(true);
      try {
        const token = await getToken();
        const educationLevel = audienceToEducation(newAudienceLevel);
        
        const response = await fetch('/api/d1/user/education-level', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ educationLevel }),
        });

        if (!response.ok) {
          throw new Error('Failed to save education level');
        }

        toast.success('Education level saved successfully!');
      } catch (error) {
        console.error('Error saving education level:', error);
        toast.error('Failed to save education level. Please try again.');
      } finally {
        setIsSubmitting(false);
      }
    }
  };

  const currentLevelData = educationLevels.find(level => level.audienceValue === currentLevel);

  if (variant === 'select') {
    return (
      <div className={`mb-6 ${className}`}>
        {showLabel && (
          <label className="block text-sm font-medium mb-2">Education Level</label>
        )}
        <select 
          value={audienceToEducation(currentLevel)}
          onChange={(e) => handleEducationLevelChange(educationToAudience(e.target.value as EducationLevel))}
          disabled={isSubmitting || disabled || isLoading}
          className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <option value="">Select your education level</option>
          {educationLevels.map((level) => (
            <option key={level.value} value={level.value}>
              {level.label}
            </option>
          ))}
        </select>
        {(isSubmitting || isLoading) && (
          <p className="text-sm text-gray-600 mt-2">
            {isLoading ? 'Loading your education level...' : 'Saving your education level...'}
          </p>
        )}
      </div>
    );
  }

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      {showLabel && (
        <>
          <GraduationCap className="h-5 w-5 text-muted-foreground" />
          <span className="text-sm text-muted-foreground">Education Level:</span>
        </>
      )}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button 
            variant="outline" 
            className="flex items-center gap-2" 
            disabled={isSubmitting || disabled || isLoading}
          >
            <span>{currentLevelData?.icon}</span>
            {isLoading ? 'Loading...' : currentLevelData?.label}
            <ChevronDown className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-48">
          {educationLevels.map((level) => (
            <DropdownMenuItem
              key={level.value}
              onClick={() => handleEducationLevelChange(level.audienceValue)}
              className={`flex items-center gap-2 ${
                currentLevel === level.audienceValue ? 'bg-accent' : ''
              }`}
              disabled={isSubmitting}
            >
              <span>{level.icon}</span>
              {level.label}
              {currentLevel === level.audienceValue && (
                <span className="ml-auto text-primary">✓</span>
              )}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
      {isSubmitting && (
        <span className="text-sm text-gray-600">Saving...</span>
      )}
    </div>
  );
} 