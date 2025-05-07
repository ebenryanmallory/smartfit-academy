import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Lessons from './pages/Lessons';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/lessons" element={<Lessons />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

function NotFound() {
  return (
    <section className="flex flex-col items-center justify-center py-24">
      <h2 className="text-3xl font-bold mb-2">404 - Not Found</h2>
      <p className="text-muted-foreground">The page you are looking for does not exist.</p>
    </section>
  );
}

export default App;
