import { useState, useRef, useEffect } from 'react';
import { useChatStore } from '@/lib/store/chatStore';
import { useDicomStore } from '@/lib/store/dicomStore';
import { useFeedbackStore } from '@/lib/store/feedbackStore';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import {
  Send,
  Paperclip,
  Bot,
  User,
  Loader2,
  ThumbsUp,
  ThumbsDown,
  Copy,
  RefreshCw,
  Image,
  FileText,
  Sparkles,
  MessageSquare,
} from 'lucide-react';
import { toast } from '@/hooks/use-toast';

export default function ChatPage() {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const {
    messages,
    isGenerating,
    sendChatMessage,
    clearMessages,
  } = useChatStore();
  const { currentStudy } = useDicomStore();
  const { addChatFeedback } = useFeedbackStore();

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isGenerating) return;

    const message = input.trim();
    setInput('');
    await sendChatMessage(message);
  };

  const handleFeedback = (messageId: string, rating: 'positive' | 'negative') => {
    addChatFeedback({
      messageId,
      sessionId: 'current-session',
      rating,
    });
    toast({
      title: 'Feedback submitted',
      description: 'Thank you for your feedback!',
    });
  };

  const copyToClipboard = (content: string) => {
    navigator.clipboard.writeText(content);
    toast({
      title: 'Copied',
      description: 'Message copied to clipboard',
    });
  };

  const suggestedPrompts = [
    'Describe the findings in this chest CT',
    'What is the differential diagnosis for this pattern?',
    'Generate a report for this study',
    'Compare with previous imaging',
  ];

  return (
    <div className="h-[calc(100vh-8rem)] flex gap-4 animate-fade-in">
      {/* Main Chat Area */}
      <Card className="flex-1 flex flex-col">
        <CardHeader className="border-b border-border pb-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-primary text-primary-foreground">
                <Bot className="w-5 h-5" />
              </div>
              <div>
                <CardTitle className="text-lg">AI Diagnostic Assistant</CardTitle>
                <p className="text-sm text-muted-foreground">
                  {currentStudy
                    ? `Analyzing: ${currentStudy.studyDescription}`
                    : 'Ready to assist with medical imaging analysis'}
                </p>
              </div>
            </div>
            <Button variant="outline" size="sm" onClick={clearMessages}>
              <RefreshCw className="w-4 h-4 mr-2" />
              New Chat
            </Button>
          </div>
        </CardHeader>

        <ScrollArea className="flex-1 p-4">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-8">
              <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-primary/10 mb-4">
                <Sparkles className="w-8 h-8 text-primary" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">
                How can I assist you today?
              </h3>
              <p className="text-muted-foreground mb-6 max-w-md">
                Ask questions about medical imaging, request diagnostic analysis,
                or get help generating reports.
              </p>
              <div className="grid grid-cols-2 gap-2 max-w-lg">
                {suggestedPrompts.map((prompt, i) => (
                  <Button
                    key={i}
                    variant="outline"
                    className="h-auto py-3 px-4 text-left justify-start"
                    onClick={() => setInput(prompt)}
                  >
                    <MessageSquare className="w-4 h-4 mr-2 flex-shrink-0" />
                    <span className="text-sm">{prompt}</span>
                  </Button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    "flex gap-3",
                    message.role === 'user' && "flex-row-reverse"
                  )}
                >
                  <div
                    className={cn(
                      "flex items-center justify-center w-8 h-8 rounded-full flex-shrink-0",
                      message.role === 'user'
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted"
                    )}
                  >
                    {message.role === 'user' ? (
                      <User className="w-4 h-4" />
                    ) : (
                      <Bot className="w-4 h-4" />
                    )}
                  </div>

                  <div
                    className={cn(
                      "flex-1 max-w-[80%]",
                      message.role === 'user' && "flex flex-col items-end"
                    )}
                  >
                    <div
                      className={cn(
                        "rounded-2xl px-4 py-3",
                        message.role === 'user'
                          ? "bg-primary text-primary-foreground rounded-tr-sm"
                          : "bg-muted rounded-tl-sm"
                      )}
                    >
                      <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                      {message.isStreaming && (
                        <span className="inline-block w-2 h-4 bg-current animate-pulse ml-1" />
                      )}
                    </div>

                    {message.role === 'assistant' && !message.isStreaming && (
                      <div className="flex items-center gap-1 mt-2">
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          onClick={() => copyToClipboard(message.content)}
                        >
                          <Copy className="w-3 h-3" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          onClick={() => handleFeedback(message.id, 'positive')}
                        >
                          <ThumbsUp className="w-3 h-3" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          onClick={() => handleFeedback(message.id, 'negative')}
                        >
                          <ThumbsDown className="w-3 h-3" />
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </ScrollArea>

        {/* Input Area */}
        <div className="p-4 border-t border-border">
          <form onSubmit={handleSubmit}>
            <div className="flex gap-2">
              <Button type="button" variant="outline" size="icon">
                <Paperclip className="w-4 h-4" />
              </Button>
              <div className="flex-1 relative">
                <Textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask about the imaging study..."
                  className="min-h-[44px] max-h-32 resize-none pr-12"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit(e);
                    }
                  }}
                />
              </div>
              <Button
                type="submit"
                variant="medical"
                size="icon"
                disabled={!input.trim() || isGenerating}
              >
                {isGenerating ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </Button>
            </div>
          </form>
        </div>
      </Card>

      {/* Context Panel */}
      <Card className="w-72 flex-shrink-0">
        <CardHeader>
          <CardTitle className="text-sm">Context</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {currentStudy ? (
            <>
              <div className="p-3 rounded-lg bg-muted/50 border border-border">
                <div className="flex items-center gap-2 mb-2">
                  <Image className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium">Active Study</span>
                </div>
                <p className="text-xs text-muted-foreground">{currentStudy.patientName}</p>
                <p className="text-xs text-muted-foreground">{currentStudy.studyDescription}</p>
                <p className="text-xs text-muted-foreground">{currentStudy.modality}</p>
                <p className="text-xs text-muted-foreground mt-2">Study Date: {currentStudy.studyDate}</p>
                {useChatStore.getState().currentContext?.segmentation && (
                  <p className="text-xs text-success mt-2">✓ Segmentation Complete</p>
                )}
                {useChatStore.getState().currentContext?.similar_cases?.length > 0 && (
                  <p className="text-xs text-primary mt-1">
                    {useChatStore.getState().currentContext.similar_cases.length} similar cases found
                  </p>
                )}
              </div>

              <Button variant="outline" size="sm" className="w-full justify-start">
                <FileText className="w-4 h-4 mr-2" />
                Generate Report
              </Button>
            </>
          ) : (
            <div className="text-center py-8 text-muted-foreground text-sm">
              <Image className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No study loaded</p>
              <p className="text-xs">Upload a study to enable context-aware assistance</p>
            </div>
          )}

          <div className="pt-4 border-t border-border">
            <h4 className="text-sm font-medium mb-3">Quick Actions</h4>
            <div className="space-y-2">
              <Button variant="outline" size="sm" className="w-full justify-start">
                Summarize Findings
              </Button>
              <Button variant="outline" size="sm" className="w-full justify-start">
                Differential Diagnosis
              </Button>
              <Button variant="outline" size="sm" className="w-full justify-start">
                Clinical Recommendations
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
