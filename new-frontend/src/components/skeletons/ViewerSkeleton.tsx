import { Skeleton } from '@/components/ui/skeleton';
import { Card } from '@/components/ui/card';

export function ViewerSkeleton() {
  return (
    <div className="h-[calc(100vh-8rem)] flex gap-4 animate-fade-in">
      {/* Left Panel - Series List */}
      <Card className="w-64 flex-shrink-0 p-4">
        <Skeleton className="h-5 w-20 mb-4" />
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="space-y-2 p-3 rounded-lg border border-border">
              <Skeleton className="aspect-square w-full rounded-md" />
              <Skeleton className="h-3 w-3/4" />
              <Skeleton className="h-3 w-1/2" />
            </div>
          ))}
        </div>
      </Card>

      {/* Main Viewport */}
      <div className="flex-1 flex flex-col gap-4">
        {/* Toolbar */}
        <Card className="p-2">
          <div className="flex items-center gap-2">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <Skeleton key={i} className="h-8 w-8" />
            ))}
            <div className="h-8 w-px bg-border mx-2" />
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-8 w-8" />
            ))}
            <div className="flex-1" />
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-8 w-8" />
            ))}
          </div>
        </Card>

        {/* Viewport Area */}
        <Card className="flex-1 overflow-hidden">
          <Skeleton className="w-full h-full rounded-lg" />
        </Card>
      </div>

      {/* Right Panel - Controls */}
      <Card className="w-72 flex-shrink-0 p-4">
        <div className="space-y-4">
          <div className="flex gap-2">
            <Skeleton className="h-9 flex-1" />
            <Skeleton className="h-9 flex-1" />
          </div>
          
          <div className="space-y-3 pt-4">
            <Skeleton className="h-4 w-24" />
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Skeleton className="w-3 h-3 rounded-full" />
                  <Skeleton className="h-4 w-16" />
                </div>
                <Skeleton className="h-5 w-10" />
              </div>
            ))}
          </div>
        </div>
      </Card>
    </div>
  );
}
