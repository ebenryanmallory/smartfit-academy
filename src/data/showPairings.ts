import { Brain, Users, Crown, LucideIcon } from 'lucide-react';

export interface ShowPairing {
  show: string;
  text: string;
  description: string;
  connection: string;
  featuredTitle: string;
  featuredDescription: string;
  featuredNote: string;
  icon: LucideIcon;
  color: string;
  textColor: string;
  status: 'available' | 'coming-soon';
  bookUrl?: string;
  promoVideoUrl?: string;
  seriesUrl?: string;
}

export const showPairings: ShowPairing[] = [
  {
    show: "Succession",
    text: "Machiavelli's The Prince",
    description: "The Roy family's brutal power dynamics are a masterclass in Machiavellian strategy",
    connection: "Every episode features characters wrestling with whether it's better to be feared or loved, how to maintain power through strategic cruelty, and the art of political manipulation.",
    featuredTitle: "The Roy Playbook Was Written 500 Years Ago",
    featuredDescription: "Every \"Boar on the Floor\" moment. Every calculated betrayal. Every speech about being a killer. The Roy family didn't invent these moves - they perfected what Machiavelli documented centuries ago.",
    featuredNote: "",
    icon: Crown,
    color: "from-amber-500/20 to-orange-500/20",
    textColor: "text-amber-600",
    status: "available",
    bookUrl: "/books/The%20Prince/",
    promoVideoUrl: "/promos/Succession.mov",
    seriesUrl: "https://www.youtube.com/show/SCjHhCKi82bnxvt_vbvIUZiA"
  },
  {
    show: "Stranger Things",
    text: "Jung's Shadow Psychology",
    description: "Every journey into the Upside Down is actually a trip into what Jung called 'the shadow'",
    connection: "The monsters aren't invading our world. They're emerging from it - the dark unconscious where our deepest fears take physical form.",
    featuredTitle: "Your Shadow Self Has Been Waiting",
    featuredDescription: "Carl Jung said we all have a shadow self. A dark reflection of everything we refuse to acknowledge about human nature. The Upside Down isn't a place - it's a psychological state.",
    featuredNote: "",
    icon: Brain,
    color: "from-purple-500/20 to-indigo-500/20",
    textColor: "text-purple-600",
    status: "coming-soon",
    seriesUrl: "https://www.netflix.com/title/80057281"
  },
  {
    show: "Squid Game",
    text: "Hobbes' Leviathan",
    description: "When 456 players signed away their human rights for a chance at money, they recreated the exact social contract Thomas Hobbes warned about",
    connection: "Thomas Hobbes said without government, life becomes 'nasty, brutish, and short.' But what happens when you sign your rights away to play the game?",
    featuredTitle: "The Childhood Game That Proves Hobbes Right",
    featuredDescription: "Welcome to the state of nature - now with better production values. Every game recreates Hobbes' warning about what happens when we abandon our humanity for survival.",
    featuredNote: "",
    icon: Users,
    color: "from-red-500/20 to-pink-500/20",
    textColor: "text-red-600",
    status: "coming-soon",
    seriesUrl: "https://www.netflix.com/title/81040344"
  },
  {
    show: "The Crown",
    text: "Sun Tzu's Art of War",
    description: "Every calculated pause, every strategic marriage, every perfectly timed public appearance - the Royal Family has been following Sun Tzu's 2,500-year-old playbook",
    connection: "Sun Tzu wrote that the supreme art of war is to subdue your enemy without fighting. Every royal smile, every calculated silence, every strategic marriage follows this principle.",
    featuredTitle: "The Royal Family's Secret War Manual",
    featuredDescription: "The crown jewels aren't just ceremonial. They're trophies from a war won through strategy, not violence. Every royal gesture follows ancient principles of winning without fighting.",
    featuredNote: "",
    icon: Crown,
    color: "from-amber-500/20 to-orange-500/20",
    textColor: "text-amber-600",
    status: "coming-soon",
    seriesUrl: "https://www.netflix.com/title/80025678"
  }
];
