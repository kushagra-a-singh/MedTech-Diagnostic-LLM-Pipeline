import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { useAuthStore } from '@/lib/store/authStore';
import { useIsMobile } from '@/hooks/useMediaQuery';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Upload,
  Eye,
  FileText,
  History,
  Settings,
  MessageSquare,
  ChevronLeft,
  ChevronRight,
  LogOut,
  Stethoscope,
  Users,
  BarChart3,
  Shield,
  HelpCircle,
  X,
} from 'lucide-react';

interface SidebarProps {
  onClose?: () => void;
}

// Base navigation items
const navItems = [
  { path: '/dashboard/upload', icon: Upload, label: 'Upload', description: 'Upload medical images' },
  { path: '/dashboard/view', icon: Eye, label: 'Viewer', description: 'View DICOM images' },
  { path: '/dashboard/chat', icon: MessageSquare, label: 'AI Assistant', description: 'Get AI analysis' },
  { path: '/dashboard/reports', icon: FileText, label: 'Reports', description: 'View and edit reports' },
  { path: '/dashboard/history', icon: History, label: 'History', description: 'Past sessions' },
];

// Admin-only items
const adminItems = [
  { path: '/dashboard/users', icon: Users, label: 'Users', description: 'Manage users' },
  { path: '/dashboard/analytics', icon: BarChart3, label: 'Analytics', description: 'View analytics' },
  { path: '/dashboard/security', icon: Shield, label: 'Security', description: 'Security settings' },
];

export function Sidebar({ onClose }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);
  const { user, logout } = useAuthStore();
  const location = useLocation();
  const isMobile = useIsMobile();

  const isAdmin = user?.role === 'admin';

  const NavItem = ({ item }: { item: typeof navItems[0] }) => {
    const isActive = location.pathname.startsWith(item.path);
    
    const content = (
      <NavLink
        to={item.path}
        onClick={isMobile ? onClose : undefined}
        className={cn(
          "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200",
          "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring",
          isActive && "bg-sidebar-accent text-sidebar-primary font-medium shadow-medical-sm",
          !isActive && "text-sidebar-foreground"
        )}
        aria-current={isActive ? 'page' : undefined}
      >
        <item.icon 
          className={cn("w-5 h-5 flex-shrink-0", isActive && "text-sidebar-primary")} 
          aria-hidden="true"
        />
        {!collapsed && (
          <span className="animate-fade-in truncate">{item.label}</span>
        )}
      </NavLink>
    );

    if (collapsed && !isMobile) {
      return (
        <Tooltip key={item.path}>
          <TooltipTrigger asChild>{content}</TooltipTrigger>
          <TooltipContent side="right" className="flex flex-col">
            <span className="font-medium">{item.label}</span>
            <span className="text-xs text-muted-foreground">{item.description}</span>
          </TooltipContent>
        </Tooltip>
      );
    }

    return <div key={item.path}>{content}</div>;
  };

  return (
    <aside
      className={cn(
        "flex flex-col h-screen bg-sidebar border-r border-sidebar-border transition-all duration-300 ease-in-out",
        collapsed && !isMobile ? "w-16" : "w-64"
      )}
      role="navigation"
      aria-label="Main navigation"
    >
      {/* Logo */}
      <div className="flex items-center justify-between h-16 px-4 border-b border-sidebar-border">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-primary text-primary-foreground shadow-medical-md">
            <Stethoscope className="w-5 h-5" aria-hidden="true" />
          </div>
          {(!collapsed || isMobile) && (
            <div className="animate-fade-in">
              <h1 className="font-semibold text-foreground">MedAI</h1>
              <p className="text-xs text-muted-foreground">Imaging Platform</p>
            </div>
          )}
        </div>
        {isMobile && (
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={onClose}
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5" />
          </Button>
        )}
      </div>

      {/* Navigation */}
      <ScrollArea className="flex-1">
        <nav className="p-3 space-y-1" aria-label="Primary navigation">
          {navItems.map((item) => (
            <NavItem key={item.path} item={item} />
          ))}
          
          {/* Admin Section */}
          {isAdmin && (
            <>
              <Separator className="my-3" />
              <div className={cn("px-3 py-1", collapsed && !isMobile && "hidden")}>
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  Admin
                </span>
              </div>
              {adminItems.map((item) => (
                <NavItem key={item.path} item={item} />
              ))}
            </>
          )}
        </nav>
      </ScrollArea>

      {/* Footer */}
      <div className="p-3 border-t border-sidebar-border space-y-2">
        {/* Settings */}
        <NavItem item={{ path: '/dashboard/settings', icon: Settings, label: 'Settings', description: 'App settings' }} />
        
        {/* Help */}
        <NavItem item={{ path: '/dashboard/help', icon: HelpCircle, label: 'Help', description: 'Get help' }} />

        <Separator className="my-2" />

        {/* User Section */}
        {user && (
          <div className={cn(
            "flex items-center gap-3 p-2 rounded-lg bg-sidebar-accent/50",
            collapsed && !isMobile && "justify-center"
          )}>
            <div 
              className="flex items-center justify-center w-8 h-8 rounded-full bg-primary text-primary-foreground text-sm font-medium"
              aria-hidden="true"
            >
              {user.name.charAt(0).toUpperCase()}
            </div>
            {(!collapsed || isMobile) && (
              <div className="flex-1 min-w-0 animate-fade-in">
                <p className="text-sm font-medium text-foreground truncate">{user.name}</p>
                <p className="text-xs text-muted-foreground capitalize">{user.role}</p>
              </div>
            )}
          </div>
        )}
        
        <div className="flex gap-2">
          {!isMobile && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size={collapsed ? "icon-sm" : "sm"}
                  onClick={() => setCollapsed(!collapsed)}
                  className="flex-shrink-0"
                  aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                >
                  {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">
                {collapsed ? 'Expand' : 'Collapse'}
              </TooltipContent>
            </Tooltip>
          )}
          {(!collapsed || isMobile) && (
            <Button
              variant="ghost"
              size="sm"
              onClick={logout}
              className="flex-1 justify-start gap-2 text-destructive hover:text-destructive hover:bg-destructive/10"
            >
              <LogOut className="w-4 h-4" aria-hidden="true" />
              Sign Out
            </Button>
          )}
        </div>
      </div>
    </aside>
  );
}
