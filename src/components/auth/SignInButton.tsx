import { SignInButton as ClerkSignInButton } from '@clerk/react';
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