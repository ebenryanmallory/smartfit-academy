import { Link } from "react-router-dom";

const Footer = () => (
  <footer className="border-t">
    <div className="container mx-auto px-4 py-4 text-center text-gray-600 flex flex-col md:flex-row items-center justify-between gap-2">
      <span>© 2025 Progressive AI Academy</span>
      <Link to="/style-guide" className="text-blue-600 hover:underline">Design System</Link>
    </div>
  </footer>
);

export default Footer;
