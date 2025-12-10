import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { cn } from '@/lib/utils';
import {
  ZoomIn,
  ZoomOut,
  Move,
  RotateCcw,
  RotateCw,
  Contrast,
  Ruler,
  Circle,
  Square,
  Maximize2,
  Grid3X3,
  Grid2X2,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  ChevronLeft,
  ChevronRight,
  FlipHorizontal,
  FlipVertical,
  Download,
  Share2,
  Crosshair,
  MousePointer,
  Scan,
  Layers,
  Settings2,
  MoreHorizontal,
  Keyboard,
} from 'lucide-react';

export type DicomTool = 
  | 'select' 
  | 'pan' 
  | 'zoom' 
  | 'window' 
  | 'length' 
  | 'ellipse' 
  | 'rectangle'
  | 'angle'
  | 'crosshair';

export type ViewLayout = '1x1' | '1x2' | '2x1' | '2x2' | 'mpr';

interface DicomToolbarProps {
  selectedTool: DicomTool;
  onToolSelect: (tool: DicomTool) => void;
  zoom: number;
  onZoomChange: (zoom: number) => void;
  currentSlice: number;
  totalSlices: number;
  onSliceChange: (slice: number) => void;
  isPlaying: boolean;
  onPlayToggle: () => void;
  layout: ViewLayout;
  onLayoutChange: (layout: ViewLayout) => void;
  onReset: () => void;
  onFlipH: () => void;
  onFlipV: () => void;
  onRotateCW: () => void;
  onRotateCCW: () => void;
  onExport: () => void;
  onShare: () => void;
  className?: string;
}

const tools: { id: DicomTool; icon: React.ElementType; label: string; shortcut: string }[] = [
  { id: 'select', icon: MousePointer, label: 'Select', shortcut: 'V' },
  { id: 'pan', icon: Move, label: 'Pan', shortcut: 'H' },
  { id: 'zoom', icon: ZoomIn, label: 'Zoom', shortcut: 'Z' },
  { id: 'window', icon: Contrast, label: 'Window/Level', shortcut: 'W' },
  { id: 'crosshair', icon: Crosshair, label: 'Crosshair', shortcut: 'C' },
];

const measurementTools: { id: DicomTool; icon: React.ElementType; label: string; shortcut: string }[] = [
  { id: 'length', icon: Ruler, label: 'Length', shortcut: 'L' },
  { id: 'ellipse', icon: Circle, label: 'Ellipse', shortcut: 'E' },
  { id: 'rectangle', icon: Square, label: 'Rectangle', shortcut: 'R' },
];

const layouts: { id: ViewLayout; icon: React.ElementType; label: string }[] = [
  { id: '1x1', icon: Maximize2, label: '1x1' },
  { id: '1x2', icon: Grid2X2, label: '1x2' },
  { id: '2x2', icon: Grid3X3, label: '2x2' },
  { id: 'mpr', icon: Layers, label: 'MPR' },
];

export function DicomToolbar({
  selectedTool,
  onToolSelect,
  zoom,
  onZoomChange,
  currentSlice,
  totalSlices,
  onSliceChange,
  isPlaying,
  onPlayToggle,
  layout,
  onLayoutChange,
  onReset,
  onFlipH,
  onFlipV,
  onRotateCW,
  onRotateCCW,
  onExport,
  onShare,
  className,
}: DicomToolbarProps) {
  return (
    <Card className={cn("p-2", className)}>
      <div className="flex items-center gap-1 flex-wrap" role="toolbar" aria-label="DICOM Viewer Toolbar">
        {/* Navigation Tools */}
        <div className="flex items-center gap-0.5" role="group" aria-label="Navigation tools">
          {tools.map((tool) => (
            <Tooltip key={tool.id}>
              <TooltipTrigger asChild>
                <Button
                  variant={selectedTool === tool.id ? 'default' : 'ghost'}
                  size="icon-sm"
                  onClick={() => onToolSelect(tool.id)}
                  aria-pressed={selectedTool === tool.id}
                  aria-label={`${tool.label} (${tool.shortcut})`}
                >
                  <tool.icon className="w-4 h-4" aria-hidden="true" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <span>{tool.label}</span>
                <kbd className="ml-2 px-1.5 py-0.5 text-xs bg-muted rounded">{tool.shortcut}</kbd>
              </TooltipContent>
            </Tooltip>
          ))}
        </div>

        <Separator orientation="vertical" className="h-6 mx-1" />

        {/* Measurement Tools */}
        <div className="flex items-center gap-0.5" role="group" aria-label="Measurement tools">
          {measurementTools.map((tool) => (
            <Tooltip key={tool.id}>
              <TooltipTrigger asChild>
                <Button
                  variant={selectedTool === tool.id ? 'default' : 'ghost'}
                  size="icon-sm"
                  onClick={() => onToolSelect(tool.id)}
                  aria-pressed={selectedTool === tool.id}
                  aria-label={`${tool.label} (${tool.shortcut})`}
                >
                  <tool.icon className="w-4 h-4" aria-hidden="true" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <span>{tool.label}</span>
                <kbd className="ml-2 px-1.5 py-0.5 text-xs bg-muted rounded">{tool.shortcut}</kbd>
              </TooltipContent>
            </Tooltip>
          ))}
        </div>

        <Separator orientation="vertical" className="h-6 mx-1" />

        {/* Zoom Controls */}
        <div className="flex items-center gap-0.5" role="group" aria-label="Zoom controls">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => onZoomChange(zoom * 0.9)}
                aria-label="Zoom out"
              >
                <ZoomOut className="w-4 h-4" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Zoom Out</TooltipContent>
          </Tooltip>
          
          <span className="text-xs w-12 text-center font-medium tabular-nums" aria-live="polite">
            {Math.round(zoom * 100)}%
          </span>
          
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => onZoomChange(zoom * 1.1)}
                aria-label="Zoom in"
              >
                <ZoomIn className="w-4 h-4" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Zoom In</TooltipContent>
          </Tooltip>
        </div>

        <Separator orientation="vertical" className="h-6 mx-1" />

        {/* Cine Playback */}
        <div className="flex items-center gap-0.5" role="group" aria-label="Playback controls">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => onSliceChange(1)}
                aria-label="Go to first slice"
              >
                <SkipBack className="w-4 h-4" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>First Slice</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => onSliceChange(Math.max(1, currentSlice - 1))}
                aria-label="Previous slice"
              >
                <ChevronLeft className="w-4 h-4" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Previous (↑)</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant={isPlaying ? 'default' : 'ghost'}
                size="icon-sm"
                onClick={onPlayToggle}
                aria-pressed={isPlaying}
                aria-label={isPlaying ? 'Pause playback' : 'Start playback'}
              >
                {isPlaying ? (
                  <Pause className="w-4 h-4" aria-hidden="true" />
                ) : (
                  <Play className="w-4 h-4" aria-hidden="true" />
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent>{isPlaying ? 'Pause' : 'Play'} (Space)</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => onSliceChange(Math.min(totalSlices, currentSlice + 1))}
                aria-label="Next slice"
              >
                <ChevronRight className="w-4 h-4" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Next (↓)</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => onSliceChange(totalSlices)}
                aria-label="Go to last slice"
              >
                <SkipForward className="w-4 h-4" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Last Slice</TooltipContent>
          </Tooltip>

          <span className="text-xs w-16 text-center font-medium tabular-nums" aria-live="polite">
            {currentSlice}/{totalSlices}
          </span>
        </div>

        <Separator orientation="vertical" className="h-6 mx-1" />

        {/* Layout */}
        <div className="flex items-center gap-0.5" role="group" aria-label="Layout options">
          {layouts.map((l) => (
            <Tooltip key={l.id}>
              <TooltipTrigger asChild>
                <Button
                  variant={layout === l.id ? 'default' : 'ghost'}
                  size="icon-sm"
                  onClick={() => onLayoutChange(l.id)}
                  aria-pressed={layout === l.id}
                  aria-label={`${l.label} layout`}
                >
                  <l.icon className="w-4 h-4" aria-hidden="true" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{l.label} Layout</TooltipContent>
            </Tooltip>
          ))}
        </div>

        <Separator orientation="vertical" className="h-6 mx-1" />

        {/* Transform */}
        <div className="flex items-center gap-0.5" role="group" aria-label="Transform options">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon-sm" onClick={onFlipH} aria-label="Flip horizontal">
                <FlipHorizontal className="w-4 h-4" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Flip Horizontal</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon-sm" onClick={onFlipV} aria-label="Flip vertical">
                <FlipVertical className="w-4 h-4" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Flip Vertical</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon-sm" onClick={onRotateCCW} aria-label="Rotate counterclockwise">
                <RotateCcw className="w-4 h-4" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Rotate CCW</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon-sm" onClick={onRotateCW} aria-label="Rotate clockwise">
                <RotateCw className="w-4 h-4" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Rotate CW</TooltipContent>
          </Tooltip>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Actions */}
        <div className="flex items-center gap-0.5" role="group" aria-label="Actions">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon-sm" onClick={onReset} aria-label="Reset viewport">
                <RotateCcw className="w-4 h-4" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Reset Viewport</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon-sm" onClick={onExport} aria-label="Export image">
                <Download className="w-4 h-4" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Export</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon-sm" onClick={onShare} aria-label="Share study">
                <Share2 className="w-4 h-4" aria-hidden="true" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Share</TooltipContent>
          </Tooltip>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon-sm" aria-label="More options">
                <MoreHorizontal className="w-4 h-4" aria-hidden="true" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem>
                <Settings2 className="w-4 h-4 mr-2" />
                Preferences
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Keyboard className="w-4 h-4 mr-2" />
                Keyboard Shortcuts
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <Scan className="w-4 h-4 mr-2" />
                Full Screen
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </Card>
  );
}
