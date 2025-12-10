import { useState, useEffect } from 'react';
import { useDicomStore } from '@/lib/store/dicomStore';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import {
  ZoomIn,
  ZoomOut,
  Move,
  RotateCcw,
  Contrast,
  Layers,
  Ruler,
  Circle,
  Square,
  Maximize2,
  Grid,
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff,
  Palette,
  Settings2,
  Download,
  Share2,
} from 'lucide-react';

export default function ViewerPage() {
  const {
    currentStudy,
    currentSeries,
    seriesList,
    segmentation,
    viewportConfig,
    measurements,
    updateViewportConfig,
    resetViewport,
    toggleSegmentationVisibility,
    updateSegmentationOpacity,
    loadStudy,
    processSegmentation,
    isLoading,
  } = useDicomStore();

  const [selectedTool, setSelectedTool] = useState<string>('pan');
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentSlice, setCurrentSlice] = useState(1);
  const [totalSlices] = useState(50);

  useEffect(() => {
    // Load mock study on mount
    if (!currentStudy) {
      loadStudy('demo-study');
      processSegmentation('demo-study');
    }
  }, []);

  const tools = [
    { id: 'pan', icon: Move, label: 'Pan' },
    { id: 'zoom', icon: ZoomIn, label: 'Zoom' },
    { id: 'window', icon: Contrast, label: 'Window/Level' },
    { id: 'length', icon: Ruler, label: 'Length' },
    { id: 'circle', icon: Circle, label: 'Ellipse' },
    { id: 'rect', icon: Square, label: 'Rectangle' },
  ];

  return (
    <div className="h-[calc(100vh-8rem)] flex gap-4 animate-fade-in">
      {/* Left Panel - Series List */}
      <Card className="w-64 flex-shrink-0">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Series</CardTitle>
        </CardHeader>
        <CardContent className="p-2">
          <ScrollArea className="h-[calc(100vh-16rem)]">
            <div className="space-y-2">
              {seriesList.length === 0 ? (
                <div className="p-4 text-center text-muted-foreground text-sm">
                  No study loaded.<br />Upload a study first.
                </div>
              ) : (
                seriesList.map((series, index) => (
                  <button
                    key={series.seriesId}
                    className={cn(
                      "w-full p-3 rounded-lg border transition-all duration-200 text-left",
                      currentSeries?.seriesId === series.seriesId
                        ? "border-primary bg-primary/5"
                        : "border-border hover:border-primary/50 hover:bg-accent/50"
                    )}
                  >
                    <div className="aspect-square bg-muted rounded-md mb-2 flex items-center justify-center">
                      <Layers className="w-8 h-8 text-muted-foreground" />
                    </div>
                    <p className="text-xs font-medium truncate">{series.seriesDescription}</p>
                    <p className="text-xs text-muted-foreground">
                      {series.instanceCount} images
                    </p>
                  </button>
                ))
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Main Viewport */}
      <div className="flex-1 flex flex-col gap-4">
        {/* Toolbar */}
        <Card className="p-2">
          <div className="flex items-center gap-2 flex-wrap">
            {/* Tools */}
            <div className="flex items-center gap-1 border-r border-border pr-2">
              {tools.map((tool) => (
                <Button
                  key={tool.id}
                  variant={selectedTool === tool.id ? 'default' : 'ghost'}
                  size="icon-sm"
                  onClick={() => setSelectedTool(tool.id)}
                  title={tool.label}
                >
                  <tool.icon className="w-4 h-4" />
                </Button>
              ))}
            </div>

            {/* Zoom Controls */}
            <div className="flex items-center gap-1 border-r border-border pr-2">
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => updateViewportConfig({ zoom: viewportConfig.zoom * 0.9 })}
              >
                <ZoomOut className="w-4 h-4" />
              </Button>
              <span className="text-xs w-12 text-center">
                {Math.round(viewportConfig.zoom * 100)}%
              </span>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => updateViewportConfig({ zoom: viewportConfig.zoom * 1.1 })}
              >
                <ZoomIn className="w-4 h-4" />
              </Button>
            </div>

            {/* Playback */}
            <div className="flex items-center gap-1 border-r border-border pr-2">
              <Button variant="ghost" size="icon-sm" onClick={() => setCurrentSlice(Math.max(1, currentSlice - 1))}>
                <ChevronLeft className="w-4 h-4" />
              </Button>
              <Button variant="ghost" size="icon-sm" onClick={() => setIsPlaying(!isPlaying)}>
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </Button>
              <Button variant="ghost" size="icon-sm" onClick={() => setCurrentSlice(Math.min(totalSlices, currentSlice + 1))}>
                <ChevronRight className="w-4 h-4" />
              </Button>
              <span className="text-xs w-16 text-center">
                {currentSlice}/{totalSlices}
              </span>
            </div>

            {/* Layout */}
            <div className="flex items-center gap-1 border-r border-border pr-2">
              <Button variant="ghost" size="icon-sm" title="1x1 Layout">
                <Maximize2 className="w-4 h-4" />
              </Button>
              <Button variant="ghost" size="icon-sm" title="Grid Layout">
                <Grid className="w-4 h-4" />
              </Button>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-1 ml-auto">
              <Button variant="ghost" size="icon-sm" onClick={resetViewport} title="Reset">
                <RotateCcw className="w-4 h-4" />
              </Button>
              <Button variant="ghost" size="icon-sm" title="Export">
                <Download className="w-4 h-4" />
              </Button>
              <Button variant="ghost" size="icon-sm" title="Share">
                <Share2 className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </Card>

        {/* Viewport Area */}
        <Card className="flex-1 overflow-hidden">
          <div className="w-full h-full bg-black rounded-lg flex items-center justify-center relative">
            {/* Placeholder for DICOM viewer */}
            <div className="text-center text-white/60">
              <Layers className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg font-medium">DICOM Viewport</p>
              <p className="text-sm">Upload a study to view images</p>
            </div>

            {/* Overlay Info */}
            <div className="absolute top-4 left-4 text-white/80 text-xs space-y-1">
              <p>Patient: {currentStudy?.patientName || 'N/A'}</p>
              <p>Study: {currentStudy?.studyDescription || 'N/A'}</p>
              <p>Slice: {currentSlice}/{totalSlices}</p>
            </div>

            <div className="absolute top-4 right-4 text-white/80 text-xs space-y-1 text-right">
              <p>WC: {viewportConfig.windowCenter}</p>
              <p>WW: {viewportConfig.windowWidth}</p>
              <p>Zoom: {Math.round(viewportConfig.zoom * 100)}%</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Right Panel - Controls */}
      <Card className="w-72 flex-shrink-0">
        <Tabs defaultValue="segmentation" className="h-full">
          <TabsList className="w-full grid grid-cols-2 m-2" style={{ width: 'calc(100% - 16px)' }}>
            <TabsTrigger value="segmentation">Overlays</TabsTrigger>
            <TabsTrigger value="window">Window</TabsTrigger>
          </TabsList>

          <TabsContent value="segmentation" className="p-4 pt-0">
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Toggle segmentation overlays
              </p>
              {segmentation.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground text-sm">
                  No segmentation available
                </div>
              ) : (
                <div className="space-y-3">
                  {segmentation.map((overlay) => (
                    <div key={overlay.id} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: overlay.color }}
                          />
                          <Label className="text-sm">{overlay.name}</Label>
                        </div>
                        <Switch
                          checked={overlay.visible}
                          onCheckedChange={() => toggleSegmentationVisibility(overlay.id)}
                        />
                      </div>
                      {overlay.visible && (
                        <div className="pl-5">
                          <div className="flex items-center gap-2">
                            <Label className="text-xs text-muted-foreground">Opacity</Label>
                            <Slider
                              value={[overlay.opacity * 100]}
                              onValueChange={([v]) => updateSegmentationOpacity(overlay.id, v / 100)}
                              max={100}
                              step={5}
                              className="flex-1"
                            />
                            <span className="text-xs w-8">{Math.round(overlay.opacity * 100)}%</span>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="window" className="p-4 pt-0">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label className="text-sm">Window Center</Label>
                <Slider
                  value={[viewportConfig.windowCenter]}
                  onValueChange={([v]) => updateViewportConfig({ windowCenter: v })}
                  min={-1000}
                  max={1000}
                  step={1}
                />
                <p className="text-xs text-muted-foreground text-right">
                  {viewportConfig.windowCenter}
                </p>
              </div>

              <div className="space-y-2">
                <Label className="text-sm">Window Width</Label>
                <Slider
                  value={[viewportConfig.windowWidth]}
                  onValueChange={([v]) => updateViewportConfig({ windowWidth: v })}
                  min={1}
                  max={2000}
                  step={1}
                />
                <p className="text-xs text-muted-foreground text-right">
                  {viewportConfig.windowWidth}
                </p>
              </div>

              <div className="pt-4 border-t border-border space-y-3">
                <Label className="text-sm">Presets</Label>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { name: 'Lung', wc: -600, ww: 1500 },
                    { name: 'Bone', wc: 400, ww: 2000 },
                    { name: 'Soft', wc: 40, ww: 400 },
                    { name: 'Brain', wc: 40, ww: 80 },
                  ].map((preset) => (
                    <Button
                      key={preset.name}
                      variant="outline"
                      size="sm"
                      onClick={() => updateViewportConfig({
                        windowCenter: preset.wc,
                        windowWidth: preset.ww,
                      })}
                    >
                      {preset.name}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="flex items-center justify-between pt-4 border-t border-border">
                <Label className="text-sm">Invert</Label>
                <Switch
                  checked={viewportConfig.invert}
                  onCheckedChange={(invert) => updateViewportConfig({ invert })}
                />
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </Card>
    </div>
  );
}
