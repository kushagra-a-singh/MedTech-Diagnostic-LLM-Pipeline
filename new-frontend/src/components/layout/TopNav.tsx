import { useLocation } from 'react-router-dom';
import { Bell, Search, Moon, Sun, Menu } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useEffect, useState } from 'react';

interface TopNavProps {
  onMenuClick?: () => void;
}

const pageTitles: Record<string, string> = {
  '/dashboard/upload': 'Upload Studies',
  '/dashboard/view': 'DICOM Viewer',
  '/dashboard/chat': 'AI Assistant',
  '/dashboard/reports': 'Reports',
  '/dashboard/history': 'History',
  '/dashboard/settings': 'Settings',
};

export function TopNav({ onMenuClick }: TopNavProps) {
  const location = useLocation();
  const [isDark, setIsDark] = useState(false);
  
  const title = pageTitles[location.pathname] || 'Dashboard';

  useEffect(() => {
    const isDarkMode = document.documentElement.classList.contains('dark');
    setIsDark(isDarkMode);
  }, []);

  const toggleTheme = () => {
    document.documentElement.classList.toggle('dark');
    setIsDark(!isDark);
  };

  return (
    <header className="flex items-center justify-between h-16 px-4 lg:px-6 bg-card border-b border-border">
      <div className="flex items-center gap-3">
        {/* Mobile Menu Button */}
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={onMenuClick}
          className="lg:hidden"
          aria-label="Toggle menu"
        >
          <Menu className="w-5 h-5" />
        </Button>
        
        <h1 className="text-lg lg:text-xl font-semibold text-foreground">{title}</h1>
      </div>

      <div className="flex items-center gap-2 lg:gap-3">
        {/* Search */}
        <div className="relative hidden md:block">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search studies, reports..."
            className="w-64 pl-9 bg-background"
          />
        </div>

        {/* Theme Toggle */}
        <Button variant="ghost" size="icon" onClick={toggleTheme} aria-label="Toggle theme">
          {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </Button>

        {/* Notifications */}
        <Button variant="ghost" size="icon" className="relative" aria-label="Notifications">
          <Bell className="w-5 h-5" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-destructive rounded-full" aria-hidden="true" />
        </Button>
      </div>
    </header>
  );
}
