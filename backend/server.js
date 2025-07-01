const express = require('express');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});
app.use('/api', limiter);

// Parsing middleware
app.use(express.json({ limit: '10mb' })); // For image uploads

// LLM Service Integration
class LLMService {
  constructor() {
    // Securely stored API keys from environment variables
    this.openaiKey = process.env.OPENAI_API_KEY;
    this.geminiKey = process.env.GEMINI_API_KEY;
    this.huggingfaceKey = process.env.HUGGINGFACE_API_KEY;
    this.anthropicKey = process.env.ANTHROPIC_API_KEY;
    
    // Default to OpenAI, but can be configured
    this.defaultProvider = process.env.DEFAULT_LLM_PROVIDER || 'openai';
  }

  async callOpenAI(prompt, imageBase64 = null) {
    const messages = [
      {
        role: "system",
        content: `You are a dermatology AI assistant for DermoAI. Provide educational information about skin conditions, always include appropriate medical disclaimers, and encourage users to seek professional medical advice. Never provide specific medical diagnoses. Always be helpful, accurate, and safety-focused.`
      }
    ];

    // Build user message
    const userMessage = {
      role: "user",
      content: imageBase64 
        ? [
            { type: "text", text: prompt },
            { 
              type: "image_url", 
              image_url: { 
                url: `data:image/jpeg;base64,${imageBase64}`,
                detail: "high" 
              } 
            }
          ]
        : prompt
    };

    messages.push(userMessage);

    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.openaiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: imageBase64 ? 'gpt-4-vision-preview' : 'gpt-4',
          messages,
          max_tokens: 800,
          temperature: 0.7
        })
      });

      if (!response.ok) {
        throw new Error(`OpenAI API error: ${response.status}`);
      }

      const data = await response.json();
      return {
        message: data.choices[0].message.content,
        provider: 'openai',
        model: imageBase64 ? 'gpt-4-vision-preview' : 'gpt-4'
      };
    } catch (error) {
      console.error('OpenAI API Error:', error);
      throw error;
    }
  }

  async callGemini(prompt, imageBase64 = null) {
    const requestBody = {
      contents: [{
        parts: [
          { text: `You are a dermatology AI assistant for DermoAI. ${prompt}` },
          ...(imageBase64 ? [{ 
            inline_data: { 
              mime_type: "image/jpeg", 
              data: imageBase64 
            } 
          }] : [])
        ]
      }],
      generationConfig: {
        maxOutputTokens: 800,
        temperature: 0.7
      }
    };

    try {
      const model = imageBase64 ? 'gemini-pro-vision' : 'gemini-pro';
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${this.geminiKey}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody)
        }
      );

      if (!response.ok) {
        throw new Error(`Gemini API error: ${response.status}`);
      }

      const data = await response.json();
      return {
        message: data.candidates[0].content.parts[0].text,
        provider: 'gemini',
        model: model
      };
    } catch (error) {
      console.error('Gemini API Error:', error);
      throw error;
    }
  }

  async callAnthropic(prompt, imageBase64 = null) {
    const messages = [
      {
        role: "user",
        content: imageBase64 
          ? [
              {
                type: "image",
                source: {
                  type: "base64",
                  media_type: "image/jpeg",
                  data: imageBase64
                }
              },
              {
                type: "text",
                text: prompt
              }
            ]
          : prompt
      }
    ];

    try {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'x-api-key': this.anthropicKey,
          'Content-Type': 'application/json',
          'anthropic-version': '2023-06-01'
        },
        body: JSON.stringify({
          model: 'claude-3-sonnet-20240229',
          max_tokens: 800,
          system: "You are a dermatology AI assistant for DermoAI. Provide educational information about skin conditions, always include appropriate medical disclaimers, and encourage users to seek professional medical advice.",
          messages
        })
      });

      if (!response.ok) {
        throw new Error(`Anthropic API error: ${response.status}`);
      }

      const data = await response.json();
      return {
        message: data.content[0].text,
        provider: 'anthropic',
        model: 'claude-3-sonnet'
      };
    } catch (error) {
      console.error('Anthropic API Error:', error);
      throw error;
    }
  }

  // Main method to get AI response
  async getResponse(prompt, imageBase64 = null, preferredProvider = null) {
    const provider = preferredProvider || this.defaultProvider;
    
    try {
      switch (provider) {
        case 'openai':
          return await this.callOpenAI(prompt, imageBase64);
        case 'gemini':
          return await this.callGemini(prompt, imageBase64);
        case 'anthropic':
          return await this.callAnthropic(prompt, imageBase64);
        default:
          throw new Error(`Unsupported provider: ${provider}`);
      }
    } catch (error) {
      // Fallback to a different provider if primary fails
      console.log(`Primary provider ${provider} failed, trying fallback...`);
      
      if (provider !== 'openai' && this.openaiKey) {
        return await this.callOpenAI(prompt, imageBase64);
      } else if (provider !== 'gemini' && this.geminiKey) {
        return await this.callGemini(prompt, imageBase64);
      } else {
        throw new Error('All LLM providers failed');
      }
    }
  }
}

// Initialize LLM service
const llmService = new LLMService();

// Fallback responses for when LLM services are unavailable
const fallbackResponses = {
  greeting: "Hello! I'm here to help with dermatology questions. I'm currently experiencing some technical issues, but I'll do my best to assist you.",
  general: "I'd love to help with your dermatology question! I'm currently having some connectivity issues with my AI services, but please feel free to ask and I'll try to provide helpful information.",
  image: "I can see you've uploaded an image for analysis. While I'm currently having some technical difficulties with my image analysis capabilities, I recommend consulting with a dermatologist for professional evaluation of any skin concerns."
};

// Main chat endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { message, image, userContext } = req.body;
    
    // Input validation
    if (!message && !image) {
      return res.status(400).json({
        error: 'Message or image is required'
      });
    }

    // Log for analytics (be careful with PII)
    console.log(`Chat request from session: ${userContext?.sessionId}`);

    // Get AI response
    let aiResponse;
    try {
      aiResponse = await llmService.getResponse(
        message || "Please analyze this image for any visible skin conditions.",
        image,
        req.headers['x-preferred-provider'] // Allow frontend to specify provider
      );
    } catch (error) {
      console.error('LLM Service Error:', error);
      
      // Use fallback response
      const fallbackKey = image ? 'image' : 
                         message.toLowerCase().includes('hello') || message.toLowerCase().includes('hi') ? 'greeting' : 
                         'general';
      
      aiResponse = {
        message: fallbackResponses[fallbackKey],
        provider: 'fallback',
        model: 'internal'
      };
    }

    // Enhance response with suggestions if appropriate
    const suggestions = generateSuggestions(message, image);

    // Response
    res.json({
      message: aiResponse.message,
      suggestions,
      metadata: {
        provider: aiResponse.provider,
        model: aiResponse.model,
        timestamp: new Date().toISOString(),
        sessionId: userContext?.sessionId
      }
    });

  } catch (error) {
    console.error('Chat API Error:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: 'I apologize, but I\'m experiencing technical difficulties. Please try again later.'
    });
  }
});

// Helper function to generate contextual suggestions
function generateSuggestions(message, hasImage) {
  const suggestions = [];
  
  if (hasImage) {
    suggestions.push(
      "What should I do next?",
      "Is this something I should worry about?",
      "When should I see a dermatologist?",
      "What are the treatment options?"
    );
  } else if (message) {
    const lowerMessage = message.toLowerCase();
    if (lowerMessage.includes('acne')) {
      suggestions.push(
        "How to prevent acne?",
        "Best skincare routine for acne?",
        "When to see a dermatologist for acne?"
      );
    } else if (lowerMessage.includes('rash')) {
      suggestions.push(
        "What causes skin rashes?",
        "How to treat a rash at home?",
        "When is a rash serious?"
      );
    } else {
      suggestions.push(
        "How can I upload an image?",
        "What conditions can you identify?",
        "Find dermatologists near me"
      );
    }
  }
  
  return suggestions.slice(0, 3); // Limit to 3 suggestions
}

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    services: {
      openai: !!process.env.OPENAI_API_KEY,
      gemini: !!process.env.GEMINI_API_KEY,
      anthropic: !!process.env.ANTHROPIC_API_KEY
    }
  });
});

// Get available LLM providers
app.get('/api/providers', (req, res) => {
  const providers = [];
  
  if (process.env.OPENAI_API_KEY) {
    providers.push({
      id: 'openai',
      name: 'OpenAI GPT-4',
      supportsImages: true,
      status: 'available'
    });
  }
  
  if (process.env.GEMINI_API_KEY) {
    providers.push({
      id: 'gemini',
      name: 'Google Gemini Pro',
      supportsImages: true,
      status: 'available'
    });
  }
  
  if (process.env.ANTHROPIC_API_KEY) {
    providers.push({
      id: 'anthropic',
      name: 'Anthropic Claude',
      supportsImages: true,
      status: 'available'
    });
  }
  
  res.json({ providers });
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  res.status(500).json({
    error: 'Something went wrong!',
    message: 'Please try again later.'
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`DermoAI API server running on port ${PORT}`);
  console.log(`Available LLM providers: ${Object.keys({
    openai: process.env.OPENAI_API_KEY,
    gemini: process.env.GEMINI_API_KEY,
    anthropic: process.env.ANTHROPIC_API_KEY
  }).filter(key => process.env[key.toUpperCase() + '_API_KEY']).join(', ')}`);
});

module.exports = app;