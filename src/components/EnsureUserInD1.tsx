import { useEffect, useRef } from "react";
import { useUser } from "@clerk/clerk-react";

/**
 * Ensures that the current Clerk-authenticated user exists in D1 DB by calling the backend init endpoint once per session.
 * Place this component inside ClerkProviderWrapper in your app root.
 */
export function EnsureUserInD1() {
  const { isSignedIn, user } = useUser();
  const hasInitialized = useRef(false);

  useEffect(() => {
    if (isSignedIn && user && !hasInitialized.current) {
      hasInitialized.current = true;
      fetch("/api/d1/user/init", {
        method: "POST",
        credentials: "include",
      });
    }
  }, [isSignedIn, user]);

  return null;
}
