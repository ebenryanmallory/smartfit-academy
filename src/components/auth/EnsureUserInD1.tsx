import { useEffect, useRef } from "react";
import { useUser, useAuth } from "@clerk/clerk-react";

/**
 * Ensures that the current Clerk-authenticated user exists in D1 DB by calling the backend init endpoint once per session.
 * Place this component inside ClerkProviderWrapper in your app root.
 */
export function EnsureUserInD1() {
  const { isSignedIn, user } = useUser();
  const { getToken } = useAuth();
  const hasInitialized = useRef(false);

  useEffect(() => {
    const initUser = async () => {
      if (isSignedIn && user && !hasInitialized.current) {
        hasInitialized.current = true;
        
        try {
          const token = await getToken();
          const response = await fetch("/api/d1/user/init", {
            method: "POST",
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
            credentials: "include",
          });
          
          if (!response.ok) {
            // User initialization failed - handle silently
            const errorData = await response.text();
          }
        } catch (error) {
          // User initialization error - handle silently
        }
      }
    };

    initUser();
  }, [isSignedIn, user, getToken]);

  return null;
}
