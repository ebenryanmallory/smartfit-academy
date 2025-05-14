import React, { useState } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';

export default function ChatAssistant() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;
    const newMessages = [...messages, { role: 'user', content: input }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('/llm/llama3', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: newMessages }),
      });
      const data = await res.json();
      if (data.error) {
        setMessages([...newMessages, { role: 'assistant', content: `Error: ${data.error}` }]);
      } else if (data.result?.response) {
        setMessages([...newMessages, { role: 'assistant', content: data.result.response }]);
      } else {
        setMessages([...newMessages, { role: 'assistant', content: 'Sorry, no response.' }]);
      }
    } catch (err) {
      setMessages([...newMessages, { role: 'assistant', content: 'Error contacting assistant.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      height: '100vh',
      zIndex: 1000,
      display: 'flex',
      flexDirection: 'row',
    }}>
      <div style={{
        width: open ? 340 : 56,
        transition: 'width 0.3s',
        background: '#fff',
        borderRight: '1px solid #ddd',
        boxShadow: open ? '2px 0 8px rgba(0,0,0,0.06)' : 'none',
        height: '100%',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        alignItems: open ? 'stretch' : 'center',
        justifyContent: 'flex-start',
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          padding: open ? '12px 16px' : '12px 4px',
          borderBottom: '1px solid #eee',
          minHeight: 56,
        }}>
          <Button
            variant="ghost"
            size="icon"
            aria-label={open ? "Collapse assistant" : "Expand assistant"}
            onClick={() => setOpen(o => !o)}
            style={{ marginRight: open ? 8 : 0 }}
          >
            <span style={{
              display: 'inline-block',
              transform: open ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s',
              fontSize: 24,
            }}>
              {/* Chevron left icon */}
              <svg width="1em" height="1em" viewBox="0 0 24 24" fill="none">
                <path d="M15 6l-6 6 6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </span>
          </Button>
          {!open && (
            <span style={{
              writingMode: 'vertical-rl',
              transform: 'rotate(180deg)',
              fontSize: 14,
              color: '#333',
              marginLeft: 4,
              letterSpacing: '0.05em',
              userSelect: 'none',
            }}>
              Ask the AI Assistant
            </span>
          )}
          {open && (
            <span style={{ fontWeight: 600, fontSize: 18, color: '#222' }}>
              AI Assistant
            </span>
          )}
        </div>
        {open && (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 12 }}>
            <Card style={{ flex: 1, overflowY: 'auto', marginBottom: 10, background: '#fafbfc' }}>
              {messages.length === 0 && (
                <div style={{ color: '#888', padding: 16, textAlign: 'center' }}>
                  Ask anything about the lesson!
                </div>
              )}
              {messages.map((msg, idx) => (
                <div key={idx} style={{
                  margin: '8px 0',
                  textAlign: msg.role === 'user' ? 'right' : 'left',
                }}>
                  <span style={{
                    display: 'inline-block',
                    background: msg.role === 'user' ? '#e0f7fa' : '#e8eaf6',
                    color: '#222',
                    borderRadius: 8,
                    padding: '6px 12px',
                    maxWidth: 220,
                    wordBreak: 'break-word',
                  }}>
                    {msg.content}
                  </span>
                </div>
              ))}
              {loading && (
                <div style={{ color: '#aaa', fontStyle: 'italic', padding: 8 }}>
                  Assistant is typing...
                </div>
              )}
            </Card>
            <div style={{ display: 'flex', gap: 8 }}>
              <Input
                placeholder="Type your question..."
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') handleSend(); }}
                disabled={loading}
                style={{ flex: 1 }}
              />
              <Button onClick={handleSend} disabled={loading || !input.trim()}>Send</Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
