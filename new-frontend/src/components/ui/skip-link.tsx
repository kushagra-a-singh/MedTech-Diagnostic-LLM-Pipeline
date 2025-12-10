import * as React from "react";
import { cn } from "@/lib/utils";

interface SkipLinkProps extends React.AnchorHTMLAttributes<HTMLAnchorElement> {
  children?: React.ReactNode;
}

/**
 * SkipLink component for keyboard navigation accessibility
 * Allows keyboard users to skip directly to main content
 */
export function SkipLink({ 
  children = "Skip to main content", 
  className,
  href = "#main-content",
  ...props 
}: SkipLinkProps) {
  return (
    <a
      href={href}
      className={cn(
        "sr-only focus:not-sr-only",
        "focus:absolute focus:top-4 focus:left-4 focus:z-50",
        "focus:px-4 focus:py-2 focus:rounded-lg",
        "focus:bg-primary focus:text-primary-foreground",
        "focus:shadow-medical-lg focus:outline-none",
        "focus:ring-2 focus:ring-ring focus:ring-offset-2",
        "transition-all duration-200",
        className
      )}
      {...props}
    >
      {children}
    </a>
  );
}
