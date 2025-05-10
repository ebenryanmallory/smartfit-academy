import React from 'react';
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Play, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import Editor from "@monaco-editor/react";

interface CodePlaygroundProps {
  initialCode: string;
  language: string;
  title?: string;
}

export function CodePlayground({ initialCode, language, title }: CodePlaygroundProps) {
  const [code, setCode] = React.useState(initialCode);
  const [output, setOutput] = React.useState<string[]>([]);
  const [isRunning, setIsRunning] = React.useState(false);

  const runCode = async () => {
    setIsRunning(true);
    setOutput([]);
    
    try {
      // For now, we'll just simulate code execution
      // In a real implementation, this would send the code to a backend service
      // that runs it in a sandbox environment
      const mockOutput = [
        "Running code...",
        "Output:",
        "Hello, World!",
        "Code execution completed."
      ];
      
      // Simulate execution time
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setOutput(mockOutput);
      toast.success("Code executed successfully!");
    } catch (error) {
      toast.error("Failed to execute code");
      setOutput(["Error: Failed to execute code"]);
    } finally {
      setIsRunning(false);
    }
  };

  const resetCode = () => {
    setCode(initialCode);
    setOutput([]);
  };

  return (
    <Card className="my-4 overflow-hidden">
      <div className="flex items-center justify-between bg-muted px-4 py-2 border-b">
        <p className="text-sm font-medium">{title || "Code Playground"}</p>
        <div className="flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="h-8"
            onClick={resetCode}
            disabled={isRunning}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Reset
          </Button>
          <Button
            variant="default"
            size="sm"
            className="h-8"
            onClick={runCode}
            disabled={isRunning}
          >
            <Play className="h-4 w-4 mr-2" />
            Run
          </Button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4">
        <div className="h-[300px] border rounded-md overflow-hidden">
          <Editor
            height="100%"
            defaultLanguage={language}
            value={code}
            onChange={(value) => setCode(value || '')}
            theme="vs-dark"
            options={{
              minimap: { enabled: false },
              fontSize: 14,
              lineNumbers: "on",
              roundedSelection: false,
              scrollBeyondLastLine: false,
              automaticLayout: true,
            }}
          />
        </div>
        
        <div className="h-[300px] border rounded-md bg-black text-white p-4 font-mono text-sm overflow-auto">
          {output.length > 0 ? (
            output.map((line, i) => (
              <div key={i} className="whitespace-pre-wrap">{line}</div>
            ))
          ) : (
            <div className="text-gray-500">Output will appear here...</div>
          )}
        </div>
      </div>
    </Card>
  );
} 