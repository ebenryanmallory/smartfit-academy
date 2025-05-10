import { Link } from 'react-router-dom';
import { Button } from '../components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '../components/ui/card';

function Home() {
  return (
    <div className="max-w-4xl mx-auto space-y-8 p-6">
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold">Welcome to Your Dashboard</h2>
        <p className="text-lg text-gray-600">
          Track your progress and continue your learning journey.
        </p>
        <Button asChild size="lg">
          <Link to="/dashboard/lessons">Continue Learning</Link>
        </Button>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Your Progress</CardTitle>
            <CardDescription>
              Track your learning journey
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-600">
              View your completed lessons, current progress, and upcoming recommendations.
            </p>
          </CardContent>
          <CardFooter>
            <Button variant="outline" asChild>
              <Link to="/dashboard/lessons">View Progress</Link>
            </Button>
          </CardFooter>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recommended Lessons</CardTitle>
            <CardDescription>
              Personalized learning path
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-600">
              Continue with your next recommended lessons based on your progress and interests.
            </p>
          </CardContent>
          <CardFooter>
            <Button variant="outline" asChild>
              <Link to="/dashboard/lessons">View Recommendations</Link>
            </Button>
          </CardFooter>
        </Card>
      </div>
    </div>
  );
}

export default Home;
