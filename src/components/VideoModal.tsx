import React, { useEffect, useRef } from 'react';
import { X } from 'lucide-react';

interface VideoModalProps {
  isOpen: boolean;
  onClose: () => void;
  videoSrc: string;
  title?: string;
}

const VideoModal: React.FC<VideoModalProps> = ({ isOpen, onClose, videoSrc, title }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
      // Focus the modal for accessibility
      modalRef.current?.focus();
    } else {
      document.body.style.overflow = 'unset';
      // Stop and reset video when modal closes
      if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.currentTime = 0;
      }
    }

    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  // Close modal on Escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
      onClick={handleBackdropClick}
      ref={modalRef}
      tabIndex={-1}
      role="dialog"
      aria-modal="true"
      aria-label={title || "Video player"}
    >
      <div className="relative w-full max-w-4xl bg-black rounded-lg overflow-hidden shadow-2xl">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-10 p-2 bg-black/50 hover:bg-black/70 text-white rounded-full transition-colors"
          aria-label="Close video"
        >
          <X className="h-5 w-5" />
        </button>

        {/* Title */}
        {title && (
          <div className="absolute top-4 left-4 z-10 bg-black/50 text-white px-3 py-1 rounded text-sm font-medium">
            {title}
          </div>
        )}

        {/* Video element */}
        <video
          ref={videoRef}
          src={videoSrc}
          controls
          autoPlay
          className="w-full h-auto max-h-[80vh] object-contain"
          onError={(e) => {
            console.error('Video failed to load:', e);
          }}
        >
          Your browser does not support the video tag.
        </video>
      </div>
    </div>
  );
};

export default VideoModal;