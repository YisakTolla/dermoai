import React, { useState, useRef, useEffect } from 'react';

const MessageCircleIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>
  </svg>
);

const XIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18"/>
    <line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
);

const SendIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="22" y1="2" x2="11" y2="13"/>
    <polygon points="22,2 15,22 11,13 2,9 22,2"/>
  </svg>
);

const BotIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="11" width="18" height="10" rx="2" ry="2"/>
    <circle cx="12" cy="5" r="2"/>
    <path d="M12 7v4"/>
    <line x1="8" y1="16" x2="8" y2="16"/>
    <line x1="16" y1="16" x2="16" y2="16"/>
  </svg>
);

const UserIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
    <circle cx="12" cy="7" r="4"/>
  </svg>
);

const ImageIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
    <circle cx="8.5" cy="8.5" r="1.5"/>
    <polyline points="21,15 16,10 5,21"/>
  </svg>
);

interface Message {
  id: number;
  text: string;
  sender: 'bot' | 'user';
  timestamp: Date;
  image?: string;
}

interface ChatResponse {
  message: string;
  confidence?: number;
  suggestions?: string[];
  error?: string;
}

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hi there! 👋 I'm your friendly AI skin health assistant. I'm here to help you understand more about skin conditions and can even take a look at photos if you'd like.\n\n📋 Quick disclaimer: This information is for educational purposes only and should not replace professional medical advice. If you have skin concerns, please consult with a qualified healthcare provider or dermatologist for proper diagnosis and treatment.\n\nNow, how can I help you today? You can:\n• Ask me about any skin condition\n• Upload a photo for analysis\n• Get simple skincare tips\n\nWhat would you like to know?",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'error'>('connected');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessageToAPI = async (message: string, imageBase64?: string): Promise<ChatResponse> => {
    try {
      setConnectionStatus('connecting');
      
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          image: imageBase64,
          timestamp: new Date().toISOString(),
          userContext: {
            sessionId: localStorage.getItem('sessionId') || generateSessionId(),
            userLocation: 'Sterling, VA' // or get from user
          }
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ChatResponse = await response.json();
      setConnectionStatus('connected');
      return data;

    } catch (error) {
      console.error('API Error:', error);
      setConnectionStatus('error');
      return {
        message: "I'm having trouble connecting to my AI services right now. Please try again in a moment.",
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  };

  const generateSessionId = (): string => {
    const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('sessionId', sessionId);
    return sessionId;
  };

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Validate file size (max 5MB)
      if (file.size > 5 * 1024 * 1024) {
        alert('Please select an image smaller than 5MB');
        return;
      }

      // Validate file type
      if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file');
        return;
      }

      const reader = new FileReader();
      reader.onload = () => {
        setSelectedImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const convertImageToBase64 = (dataUrl: string): string => {
    return dataUrl.split(',')[1]; // Remove data:image/jpeg;base64, prefix
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() && !selectedImage) return;

    const userMessage: Message = {
      id: Date.now(),
      text: inputValue || "Image analysis request",
      sender: 'user',
      timestamp: new Date(),
      image: selectedImage || undefined
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    try {
      const imageBase64 = selectedImage ? convertImageToBase64(selectedImage) : undefined;
      const apiResponse = await sendMessageToAPI(inputValue, imageBase64);
      
      const botResponse: Message = {
        id: Date.now() + 1,
        text: apiResponse.message,
        sender: 'bot',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, botResponse]);

      // Add suggestions as quick reply buttons if provided
      if (apiResponse.suggestions && apiResponse.suggestions.length > 0) {
        console.log('Suggestions:', apiResponse.suggestions);
      }

    } catch (error) {
      const errorResponse: Message = {
        id: Date.now() + 1,
        text: "I apologize, but I encountered an error processing your request. Please try again.",
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorResponse]);
    }

    setSelectedImage(null);
    setIsTyping(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return '#10b981';
      case 'connecting': return '#f59e0b';
      case 'error': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'AI Online';
      case 'connecting': return 'Connecting...';
      case 'error': return 'Connection Issue';
      default: return 'Unknown';
    }
  };

  return (
    <div style={{ position: 'fixed', bottom: '16px', right: '16px', zIndex: 1000 }}>
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          style={{
            backgroundColor: '#2563eb',
            color: 'white',
            borderRadius: '50%',
            padding: '16px',
            border: 'none',
            cursor: 'pointer',
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
            transition: 'all 0.3s ease',
            position: 'relative'
          }}
        >
          <MessageCircleIcon />
          <div style={{
            position: 'absolute',
            top: '-4px',
            right: '-4px',
            backgroundColor: getStatusColor(),
            color: 'white',
            fontSize: '10px',
            borderRadius: '50%',
            width: '20px',
            height: '20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            AI
          </div>
        </button>
      )}

      {isOpen && (
        <div style={{
          backgroundColor: 'white',
          borderRadius: '8px',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          width: '380px',
          height: '500px',
          display: 'flex',
          flexDirection: 'column',
          border: '1px solid #e5e7eb'
        }}>
          <div style={{
            backgroundColor: '#2563eb',
            color: 'white',
            padding: '16px',
            borderTopLeftRadius: '8px',
            borderTopRightRadius: '8px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <BotIcon />
              <span style={{ fontWeight: '600' }}>AI Assistant</span>
              <span style={{ 
                fontSize: '10px', 
                backgroundColor: getStatusColor(), 
                padding: '2px 6px', 
                borderRadius: '4px' 
              }}>
                {getStatusText()}
              </span>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              style={{
                backgroundColor: 'transparent',
                border: 'none',
                color: 'white',
                cursor: 'pointer',
                borderRadius: '50%',
                padding: '4px'
              }}
            >
              <XIcon />
            </button>
          </div>

          <div style={{
            flex: 1,
            overflowY: 'auto',
            padding: '16px',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px'
          }}>
            {messages.map((message) => (
              <div key={message.id} style={{
                display: 'flex',
                justifyContent: message.sender === 'user' ? 'flex-end' : 'flex-start'
              }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '8px',
                  maxWidth: '85%',
                  flexDirection: message.sender === 'user' ? 'row-reverse' : 'row'
                }}>
                  <div style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '50%',
                    backgroundColor: message.sender === 'user' ? '#10b981' : '#2563eb',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white'
                  }}>
                    {message.sender === 'user' ? <UserIcon /> : <BotIcon />}
                  </div>
                  <div style={{
                    borderRadius: '8px',
                    padding: '12px',
                    backgroundColor: message.sender === 'user' ? '#10b981' : '#f3f4f6',
                    color: message.sender === 'user' ? 'white' : '#1f2937'
                  }}>
                    {message.image && (
                      <img 
                        src={message.image} 
                        alt="User uploaded" 
                        style={{ 
                          maxWidth: '150px', 
                          maxHeight: '100px', 
                          borderRadius: '4px', 
                          marginBottom: '8px',
                          objectFit: 'cover'
                        }} 
                      />
                    )}
                    <p style={{ fontSize: '14px', margin: 0, lineHeight: '1.4', whiteSpace: 'pre-line' }}>
                      {message.text}
                    </p>
                    <span style={{ fontSize: '12px', opacity: 0.7, marginTop: '4px', display: 'block' }}>
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                </div>
              </div>
            ))}
            
            {isTyping && (
              <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                  <div style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '50%',
                    backgroundColor: '#2563eb',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white'
                  }}>
                    <BotIcon />
                  </div>
                  <div style={{ backgroundColor: '#f3f4f6', borderRadius: '8px', padding: '12px' }}>
                    <div style={{ display: 'flex', gap: '4px' }}>
                      {[0, 1, 2].map((i) => (
                        <div
                          key={i}
                          style={{
                            width: '8px',
                            height: '8px',
                            backgroundColor: '#9ca3af',
                            borderRadius: '50%',
                            animation: `bounce 1.4s infinite ${i * 0.2}s`
                          }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {selectedImage && (
            <div style={{ padding: '8px 16px', borderTop: '1px solid #e5e7eb' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <img 
                  src={selectedImage} 
                  alt="Selected" 
                  style={{ width: '40px', height: '40px', borderRadius: '4px', objectFit: 'cover' }}
                />
                <span style={{ fontSize: '12px', color: '#6b7280' }}>Image ready for analysis</span>
                <button 
                  onClick={() => setSelectedImage(null)}
                  style={{ marginLeft: 'auto', background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer' }}
                >
                  Remove
                </button>
              </div>
            </div>
          )}

          <div style={{ padding: '16px', borderTop: '1px solid #e5e7eb' }}>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-end' }}>
              <button
                onClick={() => fileInputRef.current?.click()}
                style={{
                  backgroundColor: '#f3f4f6',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  padding: '8px',
                  cursor: 'pointer'
                }}
              >
                <ImageIcon />
              </button>
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about skin conditions or upload an image..."
                style={{
                  flex: 1,
                  border: '1px solid #d1d5db',
                  borderRadius: '8px',
                  padding: '8px 12px',
                  fontSize: '14px',
                  outline: 'none'
                }}
              />
              <button
                onClick={handleSendMessage}
                disabled={(!inputValue.trim() && !selectedImage) || connectionStatus === 'connecting'}
                style={{
                  backgroundColor: (inputValue.trim() || selectedImage) && connectionStatus !== 'connecting' ? '#2563eb' : '#d1d5db',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '8px',
                  cursor: (inputValue.trim() || selectedImage) && connectionStatus !== 'connecting' ? 'pointer' : 'not-allowed'
                }}
              >
                <SendIcon />
              </button>
            </div>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageSelect}
            style={{ display: 'none' }}
          />
        </div>
      )}

      <style>
        {`
          @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
          }
        `}
      </style>
    </div>
  );
};

export default Chatbot;