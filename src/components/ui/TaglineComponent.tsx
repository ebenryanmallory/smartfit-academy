import React from 'react';
import { Sparkles } from 'lucide-react';

interface TaglineComponentProps {
  className?: string;
  variant?: 'primary' | 'white';
}

const TaglineComponent: React.FC<TaglineComponentProps> = ({ 
  className = "", 
  variant = 'primary' 
}) => {
  const baseClasses = "inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium";
  const variantClasses = variant === 'white' 
    ? "bg-white/20 text-white border border-white/30 backdrop-blur-sm" 
    : "bg-primary/10 text-primary border border-primary/20 backdrop-blur-sm";

  return (
    <div className={`${baseClasses} ${variantClasses} ${className}`}>
      <Sparkles className="h-4 w-4" />
      <span>Your 'trending' timeline was scripted years ago</span>
    </div>
  );
};

export default TaglineComponent; 