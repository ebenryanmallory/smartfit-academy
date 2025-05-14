import { SignInButton as ClerkSignInButton } from '@clerk/clerk-react';
import { Button } from '../ui/button';

export const SignInButton = () => {
  return (
    <ClerkSignInButton mode="modal">
      <Button variant="default">
        Sign In
      </Button>
    </ClerkSignInButton>
  );
}; 