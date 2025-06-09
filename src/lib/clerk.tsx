import { ClerkProvider } from '@clerk/clerk-react';
import { ReactNode } from 'react';

// Robust development environment detection
const isDevelopment = () => {
  // Check multiple indicators to ensure we're truly in development
  const isViteDev = import.meta.env.DEV;
  const isLocalhost = typeof window !== 'undefined' && 
    (window.location.hostname === 'localhost' || 
     window.location.hostname === '127.0.0.1' || 
     window.location.hostname.includes('.local'));
  const hasDevMode = import.meta.env.MODE === 'development';
  
  // Must satisfy multiple conditions to be considered development
  // Localhost check is most reliable since Vite always uses localhost
  return isViteDev && (isLocalhost || hasDevMode);
};

// Initialize Clerk with appropriate key based on environment
// Use test key only in confirmed development environment
const getClerkKey = () => {
  if (isDevelopment()) {
    const testKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY_TEST;
    const prodKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;
    
    // Use test key if available in dev, fallback to prod key
    return testKey || prodKey;
  }
  
  // Always use production key in non-dev environments
  return import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;
};

const clerkPubKey = getClerkKey();

if (!clerkPubKey) {
  throw new Error('Missing Clerk Publishable Key');
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