import { Link, useLocation } from 'react-router-dom';
import { SignedIn, SignedOut, SignInButton, UserButton } from '@clerk/clerk-react';
import {
  NavigationMenu,
  NavigationMenuList,
  NavigationMenuItem,
  NavigationMenuLink,
} from './ui/navigation-menu';
import { Button } from './ui/button';
import { LayoutDashboard, BookOpen, LogIn } from 'lucide-react';

function Navigation() {
  const location = useLocation();

  return (
    <div className="flex items-center gap-4">
      <SignedIn>
        <NavigationMenu>
          <NavigationMenuList>
            <NavigationMenuItem>
              <Link 
                to="/dashboard" 
                className={`flex items-center gap-2 block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground ${
                  location.pathname === '/dashboard' ? 'bg-accent text-accent-foreground' : ''
                }`}
              >
                <LayoutDashboard className="h-4 w-4" />
                <div className="text-sm font-medium leading-none">Dashboard</div>
              </Link>
            </NavigationMenuItem>
            <NavigationMenuItem>
              <Link 
                to="/dashboard/lessons" 
                className={`flex items-center gap-2 block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground ${
                  location.pathname.startsWith('/dashboard/lessons') ? 'bg-accent text-accent-foreground' : ''
                }`}
              >
                <BookOpen className="h-4 w-4" />
                <div className="text-sm font-medium leading-none">Lessons</div>
              </Link>
            </NavigationMenuItem>
          </NavigationMenuList>
        </NavigationMenu>
      </SignedIn>

      <div className="ml-auto">
        <SignedIn>
          <UserButton 
            afterSignOutUrl="/"
            appearance={{
              elements: {
                avatarBox: "w-10 h-10"
              }
            }}
          />
        </SignedIn>
        <SignedOut>
          <SignInButton mode="modal">
            <Button variant="default" className="flex items-center gap-2">
              <LogIn className="h-4 w-4" />
              Sign In
            </Button>
          </SignInButton>
        </SignedOut>
      </div>
    </div>
  );
}

export default Navigation; 