import { Skeleton } from '@/components/ui/skeleton';
import { Card } from '@/components/ui/card';

export function ReportSkeleton() {
  return (
    <div className="h-[calc(100vh-8rem)] flex gap-4 animate-fade-in">
      {/* Left Panel - Report Editor */}
      <div className="flex-1 flex flex-col gap-4">
        {/* Actions Bar */}
        <Card className="p-3">
          <div className="flex items-center gap-3">
            <Skeleton className="h-10 w-48" />
            <Skeleton className="h-10 w-32" />
            <div className="h-8 w-px bg-border mx-1" />
            <Skeleton className="h-10 w-20" />
            <Skeleton className="h-10 w-28" />
            <Skeleton className="h-10 w-20" />
          </div>
        </Card>

        {/* Report Content */}
        <Card className="flex-1 p-6">
          <div className="space-y-6">
            <div className="flex items-center justify-between pb-4 border-b border-border">
              <div className="space-y-2">
                <Skeleton className="h-6 w-48" />
                <Skeleton className="h-4 w-32" />
              </div>
              <Skeleton className="h-6 w-20" />
            </div>
            
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="border-l-4 border-border pl-4 py-2 space-y-2">
                <div className="flex items-center gap-2">
                  <Skeleton className="h-5 w-32" />
                  <Skeleton className="h-5 w-20 rounded-full" />
                </div>
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-4/5" />
                <Skeleton className="h-4 w-3/4" />
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Right Panel - Study Info */}
      <Card className="w-72 flex-shrink-0 p-4">
        <Skeleton className="h-5 w-28 mb-4" />
        <div className="space-y-4">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="space-y-1">
              <Skeleton className="h-3 w-16" />
              <Skeleton className="h-4 w-full" />
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
