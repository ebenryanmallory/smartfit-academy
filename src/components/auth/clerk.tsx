import { ClerkProvider } from '@clerk/clerk-react';
import { ReactNode } from 'react';

// Production version - always uses production key
const clerkPubKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

if (!clerkPubKey) {
  throw new Error('Missing Clerk Production Publishable Key: VITE_CLERK_PUBLISHABLE_KEY');
}

export const ClerkProviderWrapper = ({ children }: { children: ReactNode }) => {
  return (
    <ClerkProvider 
      publishableKey={clerkPubKey}
      appearance={{
        elements: {
          // Make the modal more friendly and less intimidating
          formButtonPrimary: "bg-primary hover:bg-primary/90",
          card: "rounded-lg shadow-lg",
          headerTitle: "text-xl font-semibold",
          headerSubtitle: "text-sm text-muted-foreground",
          // Make text more readable
          formFieldLabel: "text-sm font-medium",
          formFieldInput: "text-base",
          // Ensure good contrast for accessibility
          footerActionLink: "text-primary hover:text-primary/90",
          // Style the social buttons
          socialButtonsBlockButton: "w-full",
          socialButtonsBlockButtonText: "text-sm font-medium",
          // Style the divider
          dividerLine: "bg-border",
          dividerText: "text-muted-foreground text-sm",
        },
        // Use a friendly, educational tone
        variables: {
          colorPrimary: "#2563eb", // A friendly blue
          colorText: "#1f2937",    // Dark gray for good readability
          colorBackground: "#ffffff",
          colorInputBackground: "#f9fafb",
          colorInputText: "#1f2937",
        },
        // Simplify the layout
        layout: {
          socialButtonsPlacement: "bottom",
          socialButtonsVariant: "blockButton",
          showOptionalFields: false,
          privacyPageUrl: "/privacy",
          termsPageUrl: "/terms",
        }
      }}
    >
      {children}
    </ClerkProvider>
  );
}; 