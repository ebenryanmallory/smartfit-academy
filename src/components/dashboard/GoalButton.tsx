import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';

interface GoalButtonProps {
  title: string;
  description?: string;
  links?: Array<{
    text: string;
    url: string;
  }>;
  onClick: () => void;
}

export default function GoalButton({ title, description, links, onClick }: GoalButtonProps) {
  return (
    <Card className="max-w-sm">
      <CardHeader className="text-center pb-4">
        <CardTitle className="text-lg font-semibold">{title}</CardTitle>
      </CardHeader>
      <CardContent className="text-center space-y-4">
        {description && (
          <CardDescription className="text-sm text-muted-foreground">
            {description}
          </CardDescription>
        )}
        
        {links && links.length > 0 && (
          <div className="space-y-2">
            <div className="space-y-1">
              {links.map((link, index) => (
                <a
                  key={index}
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block text-xs text-primary hover:underline"
                >
                  {link.text}
                </a>
              ))}
            </div>
          </div>
        )}
        
        <Button 
          size="lg" 
          variant="outline"
          className="w-full text-lg px-8 py-4 h-auto"
          onClick={onClick}
        >
          {title}
        </Button>
      </CardContent>
    </Card>
  );
}