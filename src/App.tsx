import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from "sonner";
import { ClerkProviderWrapper } from './lib/clerk.tsx';
import { EnsureUserInD1 } from './components/auth/EnsureUserInD1';
import LandingPage from './pages/LandingPage';
import Home from './pages/Home';
import Lessons from './pages/Lessons';
import LessonPage from './pages/LessonPage';
import Navigation from './components/Navigation';
import './App.css';

const Onboarding = lazy(() => import('./pages/Onboarding'));
const SampleLesson = lazy(() => import('./pages/SampleLesson'));

function App() {
  return (
    <ClerkProviderWrapper>
      <EnsureUserInD1 />
      <BrowserRouter>
        <div className="min-h-screen flex flex-col">
          <Toaster />
          <header className="border-b">
            <div className="container mx-auto px-4 py-4 flex items-center justify-between">
              <h1 className="text-2xl font-bold">Progressive AI Academy</h1>
              <Navigation />
            </div>
          </header>
          
          <main className="flex-1">
            <Suspense fallback={<div>Loading...</div>}>
              <Routes>
                {/* Public routes */}
                <Route path="/" element={<LandingPage />} />
                <Route path="/onboarding" element={<Onboarding />} />
                <Route path="/sample-lesson" element={<SampleLesson />} />
                <Route path="/lessons/:id" element={<LessonPage />} />
                
                {/* Protected routes (will add auth check later) */}
                <Route path="/dashboard" element={<Home />} />
                <Route path="/dashboard/lessons" element={<Lessons />} />
                <Route path="/dashboard/lessons/:id" element={<Lessons />} />
                
                {/* 404 route */}
                <Route path="*" element={<NotFound />} />
              </Routes>
            </Suspense>
          </main>

          <footer className="border-t">
            <div className="container mx-auto px-4 py-4 text-center text-gray-600">
              © 2025 Progressive AI Academy
            </div>
          </footer>
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

