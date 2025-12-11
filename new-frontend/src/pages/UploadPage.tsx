import { useCallback, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { useDicomStore } from '@/lib/store/dicomStore';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { toast } from '@/hooks/use-toast';
import { 
  Upload, 
  FileImage, 
  CheckCircle2, 
  AlertCircle, 
  X,
  Loader2,
  FileUp,
  Info
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface UploadFile {
  file: File;
  id: string;
  status: 'pending' | 'uploading' | 'completed' | 'error';
  progress: number;
  error?: string;
}

export default function UploadPage() {
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const { uploadDicom, isLoading } = useDicomStore();
  const navigate = useNavigate();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadFile[] = acceptedFiles.map(file => ({
      file,
      id: `${file.name}-${Date.now()}`,
      status: 'pending',
      progress: 0,
    }));
    setFiles(prev => [...prev, ...newFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/dicom': ['.dcm', '.dicom'],
      'application/x-nifti': ['.nii', '.nii.gz'],
      'image/*': ['.png', '.jpg', '.jpeg'],
    },
    multiple: true,
  });

  const removeFile = (id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id));
  };

  const processUpload = async () => {
    if (files.length === 0) {
      toast({
        title: 'No files selected',
        description: 'Please select DICOM files to upload',
        variant: 'destructive',
      });
      return;
    }

    setIsProcessing(true);

    for (let i = 0; i < files.length; i++) {
      const uploadFile = files[i];
      
      setFiles(prev => prev.map(f => 
        f.id === uploadFile.id ? { ...f, status: 'uploading' } : f
      ));

      try {
        // Simulate progress
        for (let progress = 0; progress <= 100; progress += 10) {
          await new Promise(resolve => setTimeout(resolve, 100));
          setFiles(prev => prev.map(f => 
            f.id === uploadFile.id ? { ...f, progress } : f
          ));
        }

        const study = await uploadDicom(uploadFile.file);
        
        setFiles(prev => prev.map(f => 
          f.id === uploadFile.id ? { ...f, status: 'completed', progress: 100 } : f
        ));

        toast({
          title: 'Upload successful',
          description: `Study ${study.studyId} has been processed`,
        });
      } catch (error) {
        setFiles(prev => prev.map(f => 
          f.id === uploadFile.id ? { ...f, status: 'error', error: 'Upload failed' } : f
        ));
      }
    }

    setIsProcessing(false);
    
    // Navigate to viewer if all successful
    const allSuccessful = files.every(f => f.status === 'completed');
    if (allSuccessful) {
      setTimeout(() => {
        navigate('/dashboard/view');
      }, 1500);
    }
  };

  const getStatusIcon = (status: UploadFile['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="w-5 h-5 text-success" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-destructive" />;
      case 'uploading':
        return <Loader2 className="w-5 h-5 text-primary animate-spin" />;
      default:
        return <FileImage className="w-5 h-5 text-muted-foreground" />;
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      <div>
        <h2 className="text-2xl font-bold text-foreground">Upload Medical Images</h2>
        <p className="text-muted-foreground mt-1">
          Upload DICOM, NIfTI, or standard image files for AI analysis
        </p>
      </div>

      {/* Dropzone */}
      <Card variant="bordered">
        <CardContent className="p-0">
          <div
            {...getRootProps()}
            className={cn(
              "flex flex-col items-center justify-center p-12 border-2 border-dashed rounded-xl cursor-pointer transition-all duration-200",
              isDragActive 
                ? "border-primary bg-primary/5" 
                : "border-border hover:border-primary/50 hover:bg-accent/50"
            )}
          >
            <input {...getInputProps()} />
            <div className={cn(
              "flex items-center justify-center w-16 h-16 rounded-2xl mb-4 transition-all duration-200",
              isDragActive ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
            )}>
              <Upload className="w-8 h-8" />
            </div>
            <h3 className="text-lg font-semibold text-foreground mb-1">
              {isDragActive ? 'Drop files here' : 'Drag & drop files'}
            </h3>
            <p className="text-sm text-muted-foreground mb-4">
              or click to browse your computer
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              <span className="px-2 py-1 text-xs font-medium bg-primary/10 text-primary rounded-full">
                .dcm
              </span>
              <span className="px-2 py-1 text-xs font-medium bg-primary/10 text-primary rounded-full">
                .dicom
              </span>
              <span className="px-2 py-1 text-xs font-medium bg-primary/10 text-primary rounded-full">
                .nii
              </span>
              <span className="px-2 py-1 text-xs font-medium bg-primary/10 text-primary rounded-full">
                .png/.jpg
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* File List */}
      {files.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">Selected Files ({files.length})</CardTitle>
            <CardDescription>Review files before processing</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {files.map((uploadFile) => (
              <div
                key={uploadFile.id}
                className="flex items-center gap-4 p-3 rounded-lg bg-muted/50 border border-border"
              >
                {getStatusIcon(uploadFile.status)}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">
                    {uploadFile.file.name}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {(uploadFile.file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                  {uploadFile.status === 'uploading' && (
                    <Progress value={uploadFile.progress} className="mt-2 h-1" />
                  )}
                  {uploadFile.error && (
                    <p className="text-xs text-destructive mt-1">{uploadFile.error}</p>
                  )}
                </div>
                {uploadFile.status === 'pending' && (
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => removeFile(uploadFile.id)}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                )}
              </div>
            ))}

            <div className="flex gap-3 pt-3">
              <Button
                variant="outline"
                onClick={() => setFiles([])}
                disabled={isProcessing}
              >
                Clear All
              </Button>
              <Button
                variant="medical"
                onClick={processUpload}
                disabled={isProcessing || files.length === 0}
                className="flex-1"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <FileUp className="w-4 h-4" />
                    Process {files.length} file{files.length !== 1 ? 's' : ''}
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Info Card */}
      <Card variant="glass">
        <CardContent className="flex items-start gap-4 p-4">
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary/10 flex-shrink-0">
            <Info className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h4 className="font-medium text-foreground mb-1">Supported File Types</h4>
            <p className="text-sm text-muted-foreground">
              DICOM (.dcm, .dicom), NIfTI (.nii, .nii.gz), and standard images (.png, .jpg). 
              All data is processed securely and compliant with HIPAA regulations.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
