import React from 'react';
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Play, RefreshCw, Bug } from "lucide-react";
import { toast } from "sonner";
import Editor from "@monaco-editor/react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { convertPythonToJS } from "@/utils/codeConversion";

interface CodePlaygroundTabsProps {
  pythonCode: string;
  javascriptCode?: string;
  title?: string;
}

// Debug function to be exposed to window
declare global {
  interface Window {
    debugTabs: () => void;
  }
}

export function CodePlaygroundTabs({ pythonCode, javascriptCode, title }: CodePlaygroundTabsProps) {
  const [activeTab, setActiveTab] = React.useState("python");
  const [output, setOutput] = React.useState<string[]>([]);
  const [isRunning, setIsRunning] = React.useState(false);
  
  // Auto-generate JavaScript code if not provided
  const generatedJavaScriptCode = React.useMemo(() => {
    return javascriptCode || convertPythonToJS(pythonCode);
  }, [pythonCode, javascriptCode]);

  // Expose debug function to window
  React.useEffect(() => {
    window.debugTabs = () => {
      console.log('Current tab state:', {
        activeTab,
        tabsList: document.querySelector('[role="tablist"]'),
        tabTriggers: document.querySelectorAll('[role="tab"]'),
        tabContents: document.querySelectorAll('[role="tabpanel"]')
      });
    };
  }, [activeTab]);

  const runCode = async () => {
    setIsRunning(true);
    setOutput([]);
    
    try {
      if (activeTab === "javascript") {
        // Capture console.log outputs
        const logs: string[] = [];
        const originalConsoleLog = console.log;
        console.log = (...args) => {
          logs.push(args.map(arg => 
            typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
          ).join(' '));
        };

        // Create a safe evaluation environment
        const safeEval = new Function(`
          "use strict";
          ${generatedJavaScriptCode}
        `);

        // Run the code
        safeEval();
        
        // Restore console.log
        console.log = originalConsoleLog;
        
        setOutput(logs);
        toast.success("JavaScript code executed successfully!");
      } else {
        // For Python, we'll just show a message about backend service
        setOutput([
          "Running Python code requires a backend service.",
          "This feature will be available soon!",
          "Try running the JavaScript version instead."
        ]);
        toast.info("Python execution requires backend service");
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to execute code";
      toast.error("Error executing code");
      setOutput([`Error: ${errorMessage}`]);
    } finally {
      setIsRunning(false);
    }
  };

  const resetCode = () => {
    setOutput([]);
  };

  const debugTabs = () => {
    window.debugTabs();
    toast.info("Debug info logged to console");
  };

  return (
    <Card className="my-4 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b">
        <p className="text-sm font-medium">{title || "Code Playground"}</p>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            className="h-8 bg-yellow-500 hover:bg-yellow-600 text-white"
            onClick={debugTabs}
          >
            <Bug className="h-4 w-4 mr-2" />
            Debug Tabs
          </Button>
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
      
      <div className="p-4">
        <Tabs defaultValue="python" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="python">Python</TabsTrigger>
            <TabsTrigger value="javascript">JavaScript</TabsTrigger>
          </TabsList>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <div className="h-[300px] border rounded-md overflow-hidden">
              <TabsContent value="python" className="h-full m-0">
                <Editor
                  height="100%"
                  defaultLanguage="python"
                  value={pythonCode}
                  theme="vs-dark"
                  options={{
                    minimap: { enabled: false },
                    fontSize: 14,
                    lineNumbers: "on",
                    roundedSelection: false,
                    scrollBeyondLastLine: false,
                    automaticLayout: true,
                    readOnly: true,
                  }}
                />
              </TabsContent>
              <TabsContent value="javascript" className="h-full m-0">
                <Editor
                  height="100%"
                  defaultLanguage="javascript"
                  value={generatedJavaScriptCode}
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
              </TabsContent>
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
        </Tabs>
      </div>
    </Card>
  );
} 