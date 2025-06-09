import React, { useState } from 'react';

interface NotificationBannerProps {
  message?: string;
  type?: 'info' | 'warning' | 'success' | 'error';
  dismissible?: boolean;
}

const NotificationBanner: React.FC<NotificationBannerProps> = ({
  message = "🚧 We're under heavy construction! Check back soon - good things are on the way! 🚀",
  type = 'info',
  dismissible = true
}) => {
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible) return null;

  const getBackgroundClass = () => {
    switch (type) {
      case 'warning':
        return 'bg-gradient-to-r from-[var(--color-palette-1)] to-[var(--color-palette-3)]';
      case 'success':
        return 'bg-gradient-to-r from-[var(--color-palette-1)] to-[var(--color-palette-5)]';
      case 'error':
        return 'bg-gradient-to-r from-[var(--color-palette-1)] to-[var(--color-palette-2)]';
      default:
        return 'bg-gradient-to-r from-[var(--color-palette-1)] to-[var(--color-palette-4)]';
    }
  };

  const getIcon = () => {
    switch (type) {
      case 'warning':
        return (
          <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
        );
      case 'success':
        return (
          <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        );
      case 'error':
        return (
          <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        );
      default:
        return (
          <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
        );
    }
  };

  return (
    <div className={`${getBackgroundClass()} text-white`}>
      <div className="content-container mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="flex-shrink-0">
            {getIcon()}
          </div>
          <div>
            <p className="text-sm font-medium">
              {message}
            </p>
          </div>
        </div>
        {dismissible && (
          <button
            onClick={() => setIsVisible(false)}
            className="flex-shrink-0 ml-4 p-1 rounded-md hover:bg-white/10 transition-colors"
            aria-label="Dismiss notification"
          >
            <svg className="h-4 w-4 text-gray-800" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
};

export default NotificationBanner; 