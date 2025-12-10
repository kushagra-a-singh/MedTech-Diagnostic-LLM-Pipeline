import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { toast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';
import {
  ThumbsUp,
  ThumbsDown,
  MessageSquare,
  Flag,
  Send,
  X,
  CheckCircle2,
  AlertCircle,
  Loader2,
} from 'lucide-react';

export type FeedbackType = 'positive' | 'negative' | 'correction' | 'annotation';

export interface FeedbackData {
  type: FeedbackType;
  rating?: 'positive' | 'negative';
  category?: string;
  comment: string;
  targetId: string;
  targetType: 'report' | 'chat' | 'segmentation';
}

interface FeedbackPanelProps {
  targetId: string;
  targetType: 'report' | 'chat' | 'segmentation';
  onSubmit: (feedback: FeedbackData) => Promise<void>;
  onClose?: () => void;
  className?: string;
  compact?: boolean;
}

const feedbackCategories = [
  { id: 'accuracy', label: 'Accuracy', description: 'Finding accuracy issues' },
  { id: 'completeness', label: 'Completeness', description: 'Missing information' },
  { id: 'clarity', label: 'Clarity', description: 'Unclear or confusing' },
  { id: 'formatting', label: 'Formatting', description: 'Layout or structure issues' },
  { id: 'other', label: 'Other', description: 'Other feedback' },
];

export function FeedbackPanel({
  targetId,
  targetType,
  onSubmit,
  onClose,
  className,
  compact = false,
}: FeedbackPanelProps) {
  const [feedbackType, setFeedbackType] = useState<FeedbackType | null>(null);
  const [rating, setRating] = useState<'positive' | 'negative' | null>(null);
  const [category, setCategory] = useState<string>('');
  const [comment, setComment] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleQuickRating = async (quickRating: 'positive' | 'negative') => {
    setRating(quickRating);
    if (compact) {
      await submitFeedback(quickRating);
    }
  };

  const submitFeedback = async (quickRating?: 'positive' | 'negative') => {
    setIsSubmitting(true);
    try {
      await onSubmit({
        type: feedbackType || (quickRating === 'positive' ? 'positive' : 'negative'),
        rating: quickRating || rating || undefined,
        category: category || undefined,
        comment,
        targetId,
        targetType,
      });
      
      setIsSubmitted(true);
      toast({
        title: 'Feedback submitted',
        description: 'Thank you for helping improve our AI.',
      });
      
      // Reset after delay
      setTimeout(() => {
        setIsSubmitted(false);
        setFeedbackType(null);
        setRating(null);
        setCategory('');
        setComment('');
        onClose?.();
      }, 2000);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to submit feedback. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isSubmitted) {
    return (
      <div className={cn("flex items-center gap-2 p-2 text-success", className)}>
        <CheckCircle2 className="w-4 h-4" />
        <span className="text-sm">Thank you for your feedback!</span>
      </div>
    );
  }

  if (compact) {
    return (
      <div className={cn("flex items-center gap-1", className)}>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => handleQuickRating('positive')}
          disabled={isSubmitting}
          aria-label="Good response"
          className={cn(rating === 'positive' && "text-success bg-success/10")}
        >
          <ThumbsUp className="w-3.5 h-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => handleQuickRating('negative')}
          disabled={isSubmitting}
          aria-label="Poor response"
          className={cn(rating === 'negative' && "text-destructive bg-destructive/10")}
        >
          <ThumbsDown className="w-3.5 h-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => setFeedbackType('correction')}
          aria-label="Submit correction"
        >
          <MessageSquare className="w-3.5 h-3.5" />
        </Button>
      </div>
    );
  }

  return (
    <Card className={cn("animate-fade-in", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base">Provide Feedback</CardTitle>
            <CardDescription>Help us improve our AI</CardDescription>
          </div>
          {onClose && (
            <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close">
              <X className="w-4 h-4" />
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Quick Rating */}
        <div className="flex items-center gap-2">
          <Label className="text-sm text-muted-foreground">Rate this response:</Label>
          <Button
            variant={rating === 'positive' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setRating('positive')}
            className={cn(rating === 'positive' && "bg-success hover:bg-success/90")}
          >
            <ThumbsUp className="w-4 h-4 mr-1" />
            Good
          </Button>
          <Button
            variant={rating === 'negative' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setRating('negative')}
            className={cn(rating === 'negative' && "bg-destructive hover:bg-destructive/90")}
          >
            <ThumbsDown className="w-4 h-4 mr-1" />
            Poor
          </Button>
        </div>

        {/* Category */}
        {rating === 'negative' && (
          <div className="space-y-2 animate-fade-in">
            <Label className="text-sm">What could be improved?</Label>
            <RadioGroup value={category} onValueChange={setCategory} className="grid grid-cols-2 gap-2">
              {feedbackCategories.map((cat) => (
                <div key={cat.id} className="flex items-center space-x-2">
                  <RadioGroupItem value={cat.id} id={cat.id} />
                  <Label htmlFor={cat.id} className="text-sm font-normal cursor-pointer">
                    {cat.label}
                  </Label>
                </div>
              ))}
            </RadioGroup>
          </div>
        )}

        {/* Comment */}
        <div className="space-y-2">
          <Label htmlFor="feedback-comment" className="text-sm">
            Additional comments (optional)
          </Label>
          <Textarea
            id="feedback-comment"
            placeholder="Share more details about your feedback..."
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            className="min-h-[80px]"
          />
        </div>

        {/* Submit */}
        <div className="flex justify-end gap-2">
          {onClose && (
            <Button variant="ghost" onClick={onClose}>
              Cancel
            </Button>
          )}
          <Button
            variant="medical"
            onClick={() => submitFeedback()}
            disabled={isSubmitting || (!rating && !comment)}
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Submitting...
              </>
            ) : (
              <>
                <Send className="w-4 h-4" />
                Submit Feedback
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

// Inline feedback buttons for quick rating
export function QuickFeedbackButtons({
  onPositive,
  onNegative,
  onComment,
  disabled = false,
  className,
}: {
  onPositive: () => void;
  onNegative: () => void;
  onComment?: () => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <div className={cn("flex items-center gap-1", className)}>
      <Button
        variant="ghost"
        size="icon-sm"
        onClick={onPositive}
        disabled={disabled}
        aria-label="Mark as helpful"
      >
        <ThumbsUp className="w-3.5 h-3.5" />
      </Button>
      <Button
        variant="ghost"
        size="icon-sm"
        onClick={onNegative}
        disabled={disabled}
        aria-label="Mark as unhelpful"
      >
        <ThumbsDown className="w-3.5 h-3.5" />
      </Button>
      {onComment && (
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onComment}
          disabled={disabled}
          aria-label="Add comment"
        >
          <MessageSquare className="w-3.5 h-3.5" />
        </Button>
      )}
    </div>
  );
}
