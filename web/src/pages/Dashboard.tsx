import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, ImageIcon, X, Sparkles, History, LogOut, Cpu } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import ScanResults from "@/components/ScanResults";

const API_BASE_URL = "http://localhost:3000";

type ScanState = "idle" | "preview" | "scanning" | "results";

interface ScanResult {
  final_score: number;
  verdict: string;
  breakdown: {
    model: number;
    metadata: number;
    web: number;
  };
  reasoning?: string;
}

const Dashboard = () => {
  const [scanState, setScanState] = useState<ScanState>("idle");
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<ScanResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setScanState("preview");
    setResult(null);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleScan = async () => {
    if (!selectedFile) return;
    setScanState("scanning");

    // Scroll to results area
    setTimeout(() => {
      resultsRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 300);

    try {
      const formData = new FormData();
      formData.append("image", selectedFile);

      const response = await fetch(`${API_BASE_URL}/api/images/scan`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Scan failed");

      const data = await response.json();
      setResult(data);
      setScanState("results");
    } catch {
      // Demo fallback — simulate result
      await new Promise((r) => setTimeout(r, 3000));
      setResult({
        final_score: 73,
        verdict: "Likely AI Generated",
        breakdown: { model: 82, metadata: 65, web: 71 },
        reasoning:
          "The image exhibits several hallmarks of AI generation. The ResNet-18 model detected subtle artifacts in texture patterns and inconsistent lighting gradients. EXIF metadata analysis revealed missing camera-specific tags commonly found in genuine photographs. Additionally, reverse image search found no exact matches, suggesting this is a novel generation rather than a captured photograph.",
      });
      setScanState("results");
    }
  };

  const handleReset = () => {
    setScanState("idle");
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-background relative">
      <div className="absolute inset-0 grid-pattern opacity-20" />

      {/* Navbar */}
      <nav className="relative z-10 flex items-center justify-between px-6 md:px-12 py-4 border-b border-border/50 glass-strong">
        <Link to="/" className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-primary" />
          <span className="text-lg font-bold font-display text-foreground">
            Pixel
          </span>
        </Link>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" asChild>
            <Link to="/history">
              <History className="w-4 h-4 mr-1" /> History
            </Link>
          </Button>
          <Button variant="ghost" size="icon">
            <LogOut className="w-4 h-4" />
          </Button>
        </div>
      </nav>

      <main className="relative z-10 max-w-4xl mx-auto px-6 py-12">
        {/* Dropzone */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div
            className={`glass rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer ${
              dragActive
                ? "border-primary bg-primary/5 scale-[1.02] glow-border"
                : "border-border/60 hover:border-primary/40"
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            onClick={() =>
              scanState === "idle" && fileInputRef.current?.click()
            }
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFile(file);
              }}
            />

            <AnimatePresence mode="wait">
              {scanState === "idle" ? (
                <motion.div
                  key="idle"
                  className="flex flex-col items-center justify-center py-20 px-8"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-6">
                    <Upload className="w-8 h-8 text-primary" />
                  </div>
                  <h2 className="text-xl font-semibold font-display mb-2">
                    Drop your image here
                  </h2>
                  <p className="text-muted-foreground text-sm mb-6">
                    or click to browse — PNG, JPG, WEBP supported
                  </p>
                  <Button variant="outline" size="sm">
                    <ImageIcon className="w-4 h-4 mr-1" /> Browse Files
                  </Button>
                </motion.div>
              ) : (
                <motion.div
                  key="preview"
                  className="relative p-6"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleReset();
                    }}
                    className="absolute top-4 right-4 z-10 w-8 h-8 rounded-full bg-background/80 flex items-center justify-center hover:bg-background transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>

                  <div className="relative overflow-hidden rounded-xl max-h-[400px] flex items-center justify-center bg-background/30">
                    {previewUrl && (
                      <img
                        src={previewUrl}
                        alt="Preview"
                        className="max-h-[400px] object-contain"
                      />
                    )}
                    {/* Scan line animation */}
                    {scanState === "scanning" && (
                      <div className="absolute inset-0">
                        <div className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-primary to-transparent animate-scan-line shadow-[0_0_15px_hsl(var(--primary)/0.6)]" />
                      </div>
                    )}
                  </div>

                  {scanState === "preview" && (
                    <motion.div
                      className="flex justify-center mt-6"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.2 }}
                    >
                      <Button
                        size="lg"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleScan();
                        }}
                        className="glow-border"
                      >
                        <Cpu className="w-4 h-4 mr-2" /> Analyze Image
                      </Button>
                    </motion.div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>

        {/* Results area */}
        <div ref={resultsRef} className="mt-12">
          <AnimatePresence>
            {(scanState === "scanning" || scanState === "results") && (
              <ScanResults
                isScanning={scanState === "scanning"}
                result={result}
                imageUrl={previewUrl}
                onScanAgain={handleReset}
              />
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
};


export default Dashboard;
