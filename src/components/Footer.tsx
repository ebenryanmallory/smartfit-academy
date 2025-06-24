import { Link } from "react-router-dom";

const Footer = () => (
  <footer className="border-t">
    <div className="content-container mx-auto px-4 py-4 text-center text-gray-600 flex flex-col md:flex-row items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <img src="/smartfit.svg" alt="SmartFit Academy" className="h-8 w-auto" />
        <span>© 2025 SmartFit Academy</span>
      </div>
      <div className="flex gap-4">
        <Link to="/pricing" className="focus-visible:ring-1 focus-visible:ring-ring hover:underline inline-flex items-center justify-center px-4 py-2 rounded-md text-primary text-sm transition-colors underline-offset-4 whitespace-nowrap">Pricing</Link>
        <Link to="/modern-relevance" className="focus-visible:ring-1 focus-visible:ring-ring hover:underline inline-flex items-center justify-center px-4 py-2 rounded-md text-primary text-sm transition-colors underline-offset-4 whitespace-nowrap">Modern Relevance</Link>
        <Link to="/style-guide" className="focus-visible:ring-1 focus-visible:ring-ring hover:underline inline-flex items-center justify-center px-4 py-2 rounded-md text-primary text-sm transition-colors underline-offset-4 whitespace-nowrap">Design System</Link>
      </div>
    </div>
  </footer>
);

export default Footer;
