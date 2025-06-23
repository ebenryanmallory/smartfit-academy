import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { Toaster } from "sonner";
import { ClerkProviderWrapper } from './components/auth/clerk.tsx';
import { EnsureUserInD1 } from './components/auth/EnsureUserInD1';
import LandingPage from './pages/LandingPage';
import Dashboard from './pages/Dashboard';
import Lessons from './pages/Lessons';
import LessonPage from './pages/LessonPage';
import RelevanceEngine from './pages/RelevanceEngine';
import Pricing from './pages/Pricing';
import Navigation from './components/Navigation';
import Footer from './components/Footer';
import StyleGuide from './pages/StyleGuide';
import ScrollToTop from './components/ScrollToTop';
import NotificationBanner from './components/NotificationBanner';


const Onboarding = lazy(() => import('./pages/Onboarding'));
const SampleLesson = lazy(() => import('./pages/SampleLesson'));

function App() {
  return (
    <ClerkProviderWrapper>
      <EnsureUserInD1 />
      <BrowserRouter>
        <ScrollToTop />
        <div className="min-h-screen flex flex-col">
          <Toaster />
          <header className="border-b">
            <div className="content-container mx-auto px-4 py-4 flex items-center justify-between">
              <Link to="/" className="flex items-center">
                <img 
                  src="/smartfit-full.svg" 
                  alt="SmartFit Academy" 
                  className="h-12 w-auto"
                />
              </Link>
              <Navigation />
            </div>
          </header>
          
          {/* Construction notification banner */}
          <NotificationBanner />
          
          <main className="flex-1">
            <Suspense fallback={<div>Loading...</div>}>
              <Routes>
                {/* Public routes */}
                <Route path="/" element={<LandingPage />} />
                <Route path="/onboarding" element={<Onboarding />} />
                <Route path="/sample-lesson" element={<SampleLesson />} />
                <Route path="/lessons/:id" element={<LessonPage />} />
                <Route path="/pricing" element={<Pricing />} />
                <Route path="/style-guide" element={<StyleGuide />} />
                <Route path="/modern-relevance" element={<RelevanceEngine />} />
                
                {/* Protected routes (will add auth check later) */}
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/dashboard/lessons" element={<Lessons />} />
                <Route path="/dashboard/lessons/:id" element={<Lessons />} />
                
                {/* 404 route */}
                <Route path="*" element={<NotFound />} />
              </Routes>
            </Suspense>
          </main>
          
          <Footer />
        </div>
      </BrowserRouter>
    </ClerkProviderWrapper>
  );
}

function NotFound() {
  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-3xl font-bold mb-6">404 - Page Not Found</h2>
      <p className="text-lg text-gray-600">
        The page you're looking for doesn't exist.
      </p>
    </div>
  );
}

export default App;

