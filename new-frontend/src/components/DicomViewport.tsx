import { useEffect, useRef, useState } from 'react';
import cornerstone from 'cornerstone-core';
import cornerstoneWADOImageLoader from 'cornerstone-wado-image-loader';
import dicomParser from 'dicom-parser';
import { Loader2 } from 'lucide-react';

// Initialize cornerstone-wado-image-loader
cornerstoneWADOImageLoader.external.cornerstone = cornerstone;
cornerstoneWADOImageLoader.external.dicomParser = dicomParser;

// Configure image loader
cornerstoneWADOImageLoader.configure({
    beforeSend: function (xhr) {
        // Add custom headers here (e.g. auth tokens)
        // xhr.setRequestHeader('Authorization', 'Bearer ' + token);
    }
});

interface DicomViewportProps {
    imageId: string | null;
    scale?: number;
    windowWidth?: number;
    windowCenter?: number;
}

export default function DicomViewport({
    imageId,
    scale = 1.0,
    windowWidth,
    windowCenter
}: DicomViewportProps) {
    const elementRef = useRef<HTMLDivElement>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    // Initialize viewport
    useEffect(() => {
        const element = elementRef.current;
        if (!element) return;

        cornerstone.enable(element);

        return () => {
            cornerstone.disable(element);
        };
    }, []);

    // Load and display image
    useEffect(() => {
        const element = elementRef.current;
        if (!element || !imageId) return;

        const loadAndDisplayImage = async () => {
            try {
                setLoading(true);
                setError(null);

                const image = await cornerstone.loadImage(imageId);

                cornerstone.displayImage(element, image);

                // Apply initial viewport settings if provided
                const viewport = cornerstone.getViewport(element);
                if (viewport) {
                    viewport.scale = scale;
                    if (windowWidth) viewport.voi.windowWidth = windowWidth;
                    if (windowCenter) viewport.voi.windowCenter = windowCenter;
                    cornerstone.setViewport(element, viewport);
                }
            } catch (err) {
                console.error('Error loading image:', err);
                setError('Failed to load DICOM image');
            } finally {
                setLoading(false);
            }
        };

        loadAndDisplayImage();
    }, [imageId]);

    // Update viewport when props change
    useEffect(() => {
        const element = elementRef.current;
        if (!element || !imageId) return;

        try {
            const viewport = cornerstone.getViewport(element);
            if (viewport) {
                let updated = false;

                if (viewport.scale !== scale) {
                    viewport.scale = scale;
                    updated = true;
                }

                if (windowWidth && viewport.voi.windowWidth !== windowWidth) {
                    viewport.voi.windowWidth = windowWidth;
                    updated = true;
                }

                if (windowCenter && viewport.voi.windowCenter !== windowCenter) {
                    viewport.voi.windowCenter = windowCenter;
                    updated = true;
                }

                if (updated) {
                    cornerstone.setViewport(element, viewport);
                }
            }
        } catch (e) {
            // Ignore viewport update errors if image not ready
        }
    }, [scale, windowWidth, windowCenter]);

    return (
        <div className="w-full h-full relative group">
            <div
                ref={elementRef}
                className="w-full h-full bg-black overflow-hidden"
                onContextMenu={(e) => e.preventDefault()}
            />

            {loading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
                    <Loader2 className="w-8 h-8 text-primary animate-spin" />
                </div>
            )}

            {error && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-10">
                    <div className="text-red-500 font-medium text-sm">{error}</div>
                </div>
            )}
        </div>
    );
}
