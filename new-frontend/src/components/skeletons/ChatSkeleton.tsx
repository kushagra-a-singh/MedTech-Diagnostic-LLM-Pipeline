import { Skeleton } from '@/components/ui/skeleton';
import { Card } from '@/components/ui/card';

export function ChatSkeleton() {
  return (
    <div className="h-[calc(100vh-8rem)] flex gap-4 animate-fade-in">
      {/* Main Chat Area */}
      <Card className="flex-1 flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Skeleton className="w-10 h-10 rounded-xl" />
              <div className="space-y-1">
                <Skeleton className="h-5 w-40" />
                <Skeleton className="h-4 w-56" />
              </div>
            </div>
            <Skeleton className="h-9 w-24" />
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 p-4 space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className={`flex gap-3 ${i % 2 === 0 ? 'flex-row-reverse' : ''}`}>
              <Skeleton className="w-8 h-8 rounded-full flex-shrink-0" />
              <div className={`space-y-2 ${i % 2 === 0 ? 'items-end' : ''}`}>
                <Skeleton className={`h-20 ${i % 2 === 0 ? 'w-48' : 'w-64'} rounded-2xl`} />
              </div>
            </div>
          ))}
        </div>

        {/* Input Area */}
        <div className="p-4 border-t border-border">
          <div className="flex gap-2">
            <Skeleton className="h-10 w-10" />
            <Skeleton className="h-10 flex-1" />
            <Skeleton className="h-10 w-10" />
          </div>
        </div>
      </Card>

      {/* Context Panel */}
      <Card className="w-72 flex-shrink-0 p-4">
        <Skeleton className="h-5 w-16 mb-4" />
        <div className="space-y-4">
          <div className="p-3 rounded-lg border border-border space-y-2">
            <div className="flex items-center gap-2">
              <Skeleton className="w-4 h-4" />
              <Skeleton className="h-4 w-20" />
            </div>
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-3/4" />
          </div>
          
          <Skeleton className="h-9 w-full" />
          
          <div className="pt-4 border-t border-border space-y-3">
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-9 w-full" />
            <Skeleton className="h-9 w-full" />
            <Skeleton className="h-9 w-full" />
          </div>
        </div>
      </Card>
    </div>
  );
}
