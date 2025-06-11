import React, { useState, useEffect, useImperativeHandle, forwardRef } from 'react';
import { useUser, useAuth } from '@clerk/clerk-react';
import { toast } from 'sonner';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { BookOpen, X, RefreshCw } from 'lucide-react';

interface UserTopic {
  id: number;
  user_id: string;
  topic: string;
  created_at: string;
}

interface UserTopicsProps {
  onTopicClick?: (topic: string) => void;
  className?: string;
  onTopicsChange?: (topics: UserTopic[]) => void;
}

export interface UserTopicsRef {
  refreshTopics: () => Promise<void>;
  getTopics: () => UserTopic[];
}

const UserTopics = forwardRef<UserTopicsRef, UserTopicsProps>(({ onTopicClick, className = "", onTopicsChange }, ref) => {
  const { isSignedIn } = useUser();
  const { getToken } = useAuth();
  const [topics, setTopics] = useState<UserTopic[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchTopics = async () => {
    if (!isSignedIn) {
      setError('Please sign in to view your topics');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const token = await getToken();
      if (!token) {
        throw new Error('Failed to get authentication token');
      }
      
      const response = await fetch('/api/d1/user/topics', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Failed to fetch topics:', response.status, errorText);
        throw new Error(`Failed to fetch topics: ${response.status}`);
      }
      
      const data = await response.json();
      const fetchedTopics = data.topics || [];
      setTopics(fetchedTopics);
      onTopicsChange?.(fetchedTopics);
    } catch (err) {
      console.error('Error fetching topics:', err);
      setError('Failed to load your topics');
    } finally {
      setLoading(false);
    }
  };

  const removeTopic = async (topic: string) => {
    if (!isSignedIn) {
      toast.error('Please sign in to remove topics');
      return;
    }
    
    try {
      const token = await getToken();
      if (!token) {
        throw new Error('Failed to get authentication token');
      }
      
      const response = await fetch(`/api/d1/user/topics/${encodeURIComponent(topic)}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Failed to remove topic:', response.status, errorText);
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Failed to remove topic: ${response.status}`);
      }
      
      // Refresh from server instead of updating local state
      await fetchTopics();
      toast.success('Topic removed', {
        description: `"${topic}" has been removed from your topics.`,
      });
    } catch (err) {
      console.error('Error removing topic:', err);
      toast.error('Failed to remove topic');
    }
  };

  // Expose methods to parent components
  useImperativeHandle(ref, () => ({
    refreshTopics: fetchTopics,
    getTopics: () => topics,
  }));

  useEffect(() => {
    fetchTopics();
  }, [isSignedIn]);

  if (!isSignedIn) {
    return null;
  }

  if (loading) {
    return (
      <Card className={`p-4 ${className}`}>
        <div className="flex items-center gap-2 text-muted-foreground">
          <RefreshCw className="h-4 w-4 animate-spin" />
          <span>Loading your topics...</span>
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={`p-4 border-red-200 bg-red-50 ${className}`}>
        <div className="flex items-center justify-between">
          <span className="text-red-700">{error}</span>
          <Button
            variant="ghost"
            size="sm"
            onClick={fetchTopics}
            className="text-red-700 hover:text-red-800"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </Card>
    );
  }

  if (topics.length === 0) {
    return null; // Don't show anything if no topics
  }

  return (
    <section className="container-section bg-gradient-to-r from-blue-50 to-indigo-50">
      <div className="content-container">
        <h2 className="text-3xl md:text-4xl font-bold text-center mb-6 text-foreground">
          Continue Your Learning Journey
        </h2>
        <p className="text-lg text-muted-foreground text-center mb-8 max-w-2xl mx-auto">
          Pick up where you left off with your saved learning topics
        </p>
        <div className="max-w-4xl mx-auto">
          <Card className={`p-4 bg-white shadow-sm ${className}`}>
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-foreground flex items-center gap-2">
                <BookOpen className="h-4 w-4" />
                Your Learning Topics ({topics.length})
              </h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={fetchTopics}
                className="text-muted-foreground hover:text-foreground"
                title="Refresh topics"
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
            
            <div className="flex flex-wrap gap-2">
              {topics.map((topicItem) => (
                <div key={topicItem.id} className="flex items-center gap-1">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onTopicClick?.(topicItem.topic)}
                    className="text-sm px-3 py-1 h-auto hover:bg-blue-50 hover:border-blue-300"
                  >
                    {topicItem.topic}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeTopic(topicItem.topic)}
                    className="p-1 h-auto w-auto text-gray-400 hover:text-red-500"
                    title="Remove topic"
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </section>
  );
});

export default UserTopics; 