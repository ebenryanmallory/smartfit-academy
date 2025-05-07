import { ReactNode } from 'react';

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      <Header />
      <main className="flex-1 container mx-auto px-4 py-8">{children}</main>
      <Footer />
    </div>
  );
}

function Header() {
  return (
    <header className="w-full py-4 border-b bg-white/80 backdrop-blur">
      <nav className="container mx-auto flex items-center justify-between">
        <div className="text-xl font-bold tracking-tight">AI Teaching App</div>
        <ul className="flex gap-6 text-base">
          <li><a href="/" className="hover:underline">Home</a></li>
          <li><a href="/lessons" className="hover:underline">Lessons</a></li>
          <li><a href="/about" className="hover:underline">About</a></li>
        </ul>
      </nav>
    </header>
  );
}

function Footer() {
  return (
    <footer className="w-full py-4 border-t text-center text-sm text-muted-foreground bg-white/80 backdrop-blur">
      &copy; {new Date().getFullYear()} AI Teaching App. All rights reserved.
    </footer>
  );
}
