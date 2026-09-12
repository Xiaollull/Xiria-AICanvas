import { useEffect, useRef, useState } from "react";
import { renderRasterLayer } from "./viewer-editor.js";

export default function ViewerRasterLayer({ layer, onError }) {
  const canvasRef = useRef(null);
  const generation = useRef(0);
  const onErrorRef = useRef(onError);
  const [loadedImage, setLoadedImage] = useState(null);
  onErrorRef.current = onError;

  useEffect(() => {
    const token = ++generation.current;
    setLoadedImage(null);
    const image = new Image();
    image.onload = async () => {
      try { if (image.decode) await image.decode(); } catch {}
      if (generation.current !== token) return;
      setLoadedImage(image);
    };
    image.onerror = () => { if (generation.current === token) onErrorRef.current?.(new Error("图片编辑层读取失败")); };
    image.src = layer.originalUrl || layer.url;
    return () => {
      if (generation.current === token) generation.current += 1;
      image.onload = null;
      image.onerror = null;
      image.src = "";
    };
  }, [layer.originalUrl, layer.url]);

  useEffect(() => {
    if (!loadedImage || !canvasRef.current) return;
    try {
      renderRasterLayer(canvasRef.current, loadedImage, layer);
    } catch (error) {
      onErrorRef.current?.(error);
    }
  }, [loadedImage, layer.naturalWidth, layer.naturalHeight, layer.paintStrokes]);

  return <canvas ref={canvasRef} className="viewer-raster-canvas" role="img" aria-label={layer.name || "已编辑图片"} />;
}
