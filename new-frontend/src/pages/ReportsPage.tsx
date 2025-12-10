import { useState } from 'react';
import { useReportStore } from '@/lib/store/reportStore';
import { useDicomStore } from '@/lib/store/dicomStore';
import { useFeedbackStore } from '@/lib/store/feedbackStore';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import { toast } from '@/hooks/use-toast';
import {
  FileText,
  Sparkles,
  Save,
  Download,
  Printer,
  Send,
  Edit3,
  Check,
  X,
  Plus,
  Trash2,
  GripVertical,
  Loader2,
  Copy,
  Clock,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react';

export default function ReportsPage() {
  const {
    currentReport,
    reportTemplates,
    isGenerating,
    isSaving,
    generateReport,
    saveReport,
    updateReportSection,
    addReportSection,
    removeReportSection,
  } = useReportStore();
  const { currentStudy } = useDicomStore();
  const { addReportFeedback } = useFeedbackStore();

  const [selectedTemplate, setSelectedTemplate] = useState<string>('ct-chest');
  const [editingSection, setEditingSection] = useState<string | null>(null);
  const [editContent, setEditContent] = useState('');

  const handleGenerate = async () => {
    if (!currentStudy) {
      toast({
        title: 'No study selected',
        description: 'Please load a study before generating a report',
        variant: 'destructive',
      });
      return;
    }
    await generateReport(currentStudy.studyId, selectedTemplate);
    toast({
      title: 'Report generated',
      description: 'AI has generated the report. Please review and edit as needed.',
    });
  };

  const handleSave = async () => {
    await saveReport();
    toast({
      title: 'Report saved',
      description: 'Your changes have been saved successfully',
    });
  };

  const startEditing = (sectionId: string, content: string) => {
    setEditingSection(sectionId);
    setEditContent(content);
  };

  const saveEdit = (sectionId: string, originalContent: string) => {
    if (editContent !== originalContent) {
      addReportFeedback({
        reportId: currentReport?.id || '',
        sectionId,
        originalText: originalContent,
        correctedText: editContent,
        submittedBy: 'current-user',
      });
    }
    updateReportSection(sectionId, editContent);
    setEditingSection(null);
  };

  const cancelEdit = () => {
    setEditingSection(null);
    setEditContent('');
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'draft':
        return <Badge variant="secondary"><Edit3 className="w-3 h-3 mr-1" />Draft</Badge>;
      case 'pending_review':
        return <Badge variant="outline" className="text-warning border-warning"><Clock className="w-3 h-3 mr-1" />Pending Review</Badge>;
      case 'finalized':
        return <Badge variant="outline" className="text-success border-success"><CheckCircle2 className="w-3 h-3 mr-1" />Finalized</Badge>;
      default:
        return null;
    }
  };

  return (
    <div className="h-[calc(100vh-8rem)] flex gap-4 animate-fade-in">
      {/* Left Panel - Report Editor */}
      <div className="flex-1 flex flex-col gap-4">
        {/* Actions Bar */}
        <Card className="p-3">
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex-1 min-w-[200px]">
              <Select value={selectedTemplate} onValueChange={setSelectedTemplate}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select template" />
                </SelectTrigger>
                <SelectContent>
                  {reportTemplates.map((template) => (
                    <SelectItem key={template.id} value={template.id}>
                      {template.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <Button
              variant="medical"
              onClick={handleGenerate}
              disabled={isGenerating || !currentStudy}
            >
              {isGenerating ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Generate Report
                </>
              )}
            </Button>

            <Separator orientation="vertical" className="h-8" />

            <Button variant="outline" onClick={handleSave} disabled={!currentReport || isSaving}>
              {isSaving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
              Save
            </Button>

            <Button variant="outline" disabled={!currentReport}>
              <Download className="w-4 h-4" />
              Export PDF
            </Button>

            <Button variant="outline" disabled={!currentReport}>
              <Printer className="w-4 h-4" />
              Print
            </Button>
          </div>
        </Card>

        {/* Report Content */}
        <Card className="flex-1 flex flex-col">
          {!currentReport ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
              <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-primary/10 mb-4">
                <FileText className="w-8 h-8 text-primary" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">
                No Report Generated
              </h3>
              <p className="text-muted-foreground mb-4 max-w-md">
                Select a template and click "Generate Report" to create an AI-assisted diagnostic report.
              </p>
              {!currentStudy && (
                <p className="text-sm text-warning flex items-center gap-2">
                  <AlertCircle className="w-4 h-4" />
                  Load a study first to generate a report
                </p>
              )}
            </div>
          ) : (
            <>
              <CardHeader className="border-b border-border">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>{currentReport.title}</CardTitle>
                    <CardDescription>
                      Created {new Date(currentReport.createdAt).toLocaleDateString()}
                    </CardDescription>
                  </div>
                  {getStatusBadge(currentReport.status)}
                </div>
              </CardHeader>

              <ScrollArea className="flex-1">
                <CardContent className="pt-6 space-y-6">
                  {currentReport.sections
                    .sort((a, b) => a.order - b.order)
                    .map((section) => (
                      <div
                        key={section.id}
                        className={cn(
                          "group border-l-4 pl-4 py-2 transition-all duration-200",
                          editingSection === section.id
                            ? "border-primary bg-primary/5 rounded-r-lg"
                            : section.isAIGenerated && !section.isEdited
                            ? "border-primary/30 hover:border-primary"
                            : "border-border hover:border-primary/50"
                        )}
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <h4 className="font-semibold text-foreground">
                              {section.title}
                            </h4>
                            {section.isAIGenerated && !section.isEdited && (
                              <Badge variant="secondary" className="text-xs">
                                <Sparkles className="w-3 h-3 mr-1" />
                                AI Generated
                              </Badge>
                            )}
                            {section.isEdited && (
                              <Badge variant="outline" className="text-xs">
                                <Edit3 className="w-3 h-3 mr-1" />
                                Edited
                              </Badge>
                            )}
                          </div>

                          {editingSection !== section.id && (
                            <div className="opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
                              <Button
                                variant="ghost"
                                size="icon-sm"
                                onClick={() => startEditing(section.id, section.content)}
                              >
                                <Edit3 className="w-3 h-3" />
                              </Button>
                              <Button
                                variant="ghost"
                                size="icon-sm"
                                onClick={() => navigator.clipboard.writeText(section.content)}
                              >
                                <Copy className="w-3 h-3" />
                              </Button>
                            </div>
                          )}
                        </div>

                        {editingSection === section.id ? (
                          <div className="space-y-2">
                            <Textarea
                              value={editContent}
                              onChange={(e) => setEditContent(e.target.value)}
                              className="min-h-[120px]"
                            />
                            <div className="flex gap-2">
                              <Button
                                size="sm"
                                onClick={() => saveEdit(section.id, section.content)}
                              >
                                <Check className="w-3 h-3 mr-1" />
                                Save
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={cancelEdit}
                              >
                                <X className="w-3 h-3 mr-1" />
                                Cancel
                              </Button>
                            </div>
                          </div>
                        ) : (
                          <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                            {section.content}
                          </p>
                        )}
                      </div>
                    ))}

                  <Button
                    variant="outline"
                    className="w-full border-dashed"
                    onClick={() => addReportSection({
                      title: 'New Section',
                      content: '',
                      order: currentReport.sections.length + 1,
                      isAIGenerated: false,
                      isEdited: false,
                    })}
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Add Section
                  </Button>
                </CardContent>
              </ScrollArea>
            </>
          )}
        </Card>
      </div>

      {/* Right Panel - Study Info */}
      <Card className="w-72 flex-shrink-0">
        <CardHeader>
          <CardTitle className="text-sm">Study Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {currentStudy ? (
            <>
              <div className="space-y-3">
                <div>
                  <Label className="text-xs text-muted-foreground">Patient</Label>
                  <p className="text-sm font-medium">{currentStudy.patientName}</p>
                </div>
                <div>
                  <Label className="text-xs text-muted-foreground">Patient ID</Label>
                  <p className="text-sm font-medium">{currentStudy.patientId}</p>
                </div>
                <div>
                  <Label className="text-xs text-muted-foreground">Study Date</Label>
                  <p className="text-sm font-medium">{currentStudy.studyDate}</p>
                </div>
                <div>
                  <Label className="text-xs text-muted-foreground">Modality</Label>
                  <p className="text-sm font-medium">{currentStudy.modality}</p>
                </div>
                <div>
                  <Label className="text-xs text-muted-foreground">Description</Label>
                  <p className="text-sm font-medium">{currentStudy.studyDescription}</p>
                </div>
              </div>

              <Separator />

              <div className="space-y-2">
                <Label className="text-xs text-muted-foreground">Images</Label>
                <p className="text-sm">
                  {currentStudy.seriesCount} series, {currentStudy.instanceCount} instances
                </p>
              </div>
            </>
          ) : (
            <div className="text-center py-8 text-muted-foreground text-sm">
              <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No study loaded</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
