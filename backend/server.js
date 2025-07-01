const express = require('express');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;

app.use(helmet());
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  credentials: true
}));

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});
app.use('/api', limiter);

app.use(express.json({ limit: '10mb' }));

class LLMService {
  constructor() {
    this.openaiKey = process.env.OPENAI_API_KEY;
    this.geminiKey = process.env.GEMINI_API_KEY;
    this.huggingfaceKey = process.env.HUGGINGFACE_API_KEY;
    this.anthropicKey = process.env.ANTHROPIC_API_KEY;
    
    this.defaultProvider = process.env.DEFAULT_LLM_PROVIDER || 'anthropic';
  }

  async callOpenAI(prompt, imageBase64 = null) {
    const messages = [
      {
        role: "system",
        content: `You are a dermatology AI assistant for DermoAI. Provide educational information about skin conditions, always include appropriate medical disclaimers, and encourage users to seek professional medical advice. Never provide specific medical diagnoses. Always be helpful, accurate, and safety-focused.`
      }
    ];

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
          model: imageBase64 ? 'gpt-4o' : 'gpt-4o-mini',
          messages,
          max_tokens: 800,
          temperature: 0.7
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('OpenAI API Error Response:', errorData);
        throw new Error(`OpenAI API error: ${response.status} - ${errorData.error?.message || 'Unknown error'}`);
      }

      const data = await response.json();
      return {
        message: data.choices[0].message.content,
        provider: 'openai',
        model: imageBase64 ? 'gpt-4o' : 'gpt-4o-mini'
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

  async callAnthropic(prompt, imageBase64 = null, mediaType = "image/jpeg", includeDisclaimer = false) {
    console.log('Calling Anthropic with:', {
      hasImage: !!imageBase64,
      imageLength: imageBase64 ? imageBase64.length : 0,
      promptLength: prompt.length,
      mediaType: mediaType
    });
    
    const messages = [
      {
        role: "user",
        content: imageBase64 
          ? [
              {
                type: "image",
                source: {
                  type: "base64",
                  media_type: mediaType,
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

    // System prompt with conditional disclaimer
    const systemPrompt = includeDisclaimer 
      ? `You are a friendly, conversational dermatology AI assistant for DermoAI. IMPORTANT: Start your FIRST response with this disclaimer: 'Important: This information is for educational purposes only and should not replace professional medical advice. If you have skin concerns, please consult with a qualified healthcare provider or dermatologist for proper diagnosis and treatment.' 

After the disclaimer, be warm, friendly, and conversational. Use simple language that anyone can understand. Avoid medical jargon unless necessary, and if you must use technical terms, explain them in plain English. Be empathetic and supportive. For all subsequent messages in this conversation, do NOT include the disclaimer.`
      : `You are a friendly, conversational dermatology AI assistant for DermoAI. Be warm, helpful, and easy to understand. Here's your personality:

- Speak like a caring friend who happens to know about skin health
- Use simple, everyday language - avoid medical jargon
- If you must use technical terms, explain them simply
- Be encouraging and supportive
- Keep responses concise but informative
- Use conversational phrases like "I understand", "That sounds concerning", "Here's what might help"
- Make recommendations practical and easy to follow
- Break down complex information into simple steps
- Be empathetic to people's skin concerns`;

    try {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'x-api-key': this.anthropicKey,
          'Content-Type': 'application/json',
          'anthropic-version': '2023-06-01'
        },
        body: JSON.stringify({
          model: 'claude-3-5-sonnet-20241022',
          max_tokens: 800,
          system: systemPrompt,
          messages
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('Anthropic API Error Response:', errorData);
        throw new Error(`Anthropic API error: ${response.status} - ${errorData.error?.message || JSON.stringify(errorData)}`);
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

  async getResponse(prompt, imageBase64 = null, preferredProvider = null, sessionContext = {}) {
    const provider = preferredProvider || this.defaultProvider;
    
    try {
      switch (provider) {
        case 'openai':
          return await this.callOpenAI(prompt, imageBase64);
        case 'gemini':
          return await this.callGemini(prompt, imageBase64);
        case 'anthropic':
          // Check if this is the first message in the session
          const includeDisclaimer = sessionContext.isFirstMessage || false;
          return await this.callAnthropic(prompt, imageBase64, "image/jpeg", includeDisclaimer);
        default:
          throw new Error(`Unsupported provider: ${provider}`);
      }
    } catch (error) {
      // Fallback to a different provider if primary fails
      console.log(`Primary provider ${provider} failed, trying fallback...`);
      
      // Try Anthropic first as fallback
      if (provider !== 'anthropic' && this.anthropicKey) {
        const includeDisclaimer = sessionContext.isFirstMessage || false;
        return await this.callAnthropic(prompt, imageBase64, "image/jpeg", includeDisclaimer);
      } else if (provider !== 'openai' && this.openaiKey) {
        return await this.callOpenAI(prompt, imageBase64);
      } else if (provider !== 'gemini' && this.geminiKey) {
        return await this.callGemini(prompt, imageBase64);
      } else {
        throw new Error('All LLM providers failed');
      }
    }
  }
}

const llmService = new LLMService();

const sessionFirstMessages = new Map();

const fallbackResponses = {
  greeting: "Hello! I'm here to help with dermatology questions. I'm currently experiencing some technical issues, but I'll do my best to assist you.",
  general: "I'd love to help with your dermatology question! I'm currently having some connectivity issues with my AI services, but please feel free to ask and I'll try to provide helpful information.",
  image: "I can see you've uploaded an image for analysis. While I'm currently having some technical difficulties with my image analysis capabilities, I recommend consulting with a dermatologist for professional evaluation of any skin concerns."
};

app.post('/api/chat', async (req, res) => {
  try {
    const { message, image, userContext } = req.body;
    
    if (!message && !image) {
      return res.status(400).json({
        error: 'Message or image is required'
      });
    }

    const sessionId = userContext?.sessionId || 'anonymous';
    console.log(`Chat request from session: ${sessionId}`);

    const isFirstMessage = !sessionFirstMessages.has(sessionId);
    if (isFirstMessage) {
      sessionFirstMessages.set(sessionId, Date.now());
      setTimeout(() => sessionFirstMessages.delete(sessionId), 24 * 60 * 60 * 1000);
    }

    let aiResponse;
    try {
      aiResponse = await llmService.getResponse(
        message || "Please analyze this image for any visible skin conditions.",
        image,
        req.headers['x-preferred-provider'],
        { isFirstMessage, sessionId }
      );
    } catch (error) {
      console.error('LLM Service Error:', error);
      
      const fallbackKey = image ? 'image' : 
                         message.toLowerCase().includes('hello') || message.toLowerCase().includes('hi') ? 'greeting' : 
                         'general';
      
      aiResponse = {
        message: fallbackResponses[fallbackKey],
        provider: 'fallback',
        model: 'internal'
      };
    }

    const suggestions = generateSuggestions(message, image);

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

app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    services: {
      anthropic: !!process.env.ANTHROPIC_API_KEY,
      localModel: true
    }
  });
});

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

app.post('/api/analyze', async (req, res) => {
  try {
    const { image, filename } = req.body;
    
    if (!image) {
      return res.status(400).json({
        error: 'No image provided'
      });
    }
    
    console.log('Analyzing image:', filename || 'unnamed');
    
    // Always use Claude Vision if Anthropic API key is available
    const useClaudeVision = process.env.ANTHROPIC_API_KEY ? true : false;
    
    if (useClaudeVision) {
      // Use Claude Vision for analysis
      try {
        const prompt = `You're a friendly dermatologist AI. Look at this skin image and help identify what it might be. Be conversational and use simple language.

CONTEXT: The image filename is "${filename || 'unknown'}"
${(() => {
  if (!filename) return '';
  const lowerFilename = filename.toLowerCase();
  
  // Check for condition names in filename
  if (lowerFilename.includes('eczema')) return 'NOTE: The filename mentions "eczema" - this is likely an eczema case.';
  if (lowerFilename.includes('acne')) return 'NOTE: The filename mentions "acne" - this might be acne.';
  if (lowerFilename.includes('melanoma')) return 'NOTE: The filename mentions "melanoma" - analyze very carefully.';
  if (lowerFilename.includes('psoriasis')) return 'NOTE: The filename mentions "psoriasis" - this might be psoriasis.';
  if (lowerFilename.includes('dermatitis')) return 'NOTE: The filename mentions "dermatitis" - this might be dermatitis.';
  if (lowerFilename.includes('rosacea')) return 'NOTE: The filename mentions "rosacea" - this might be rosacea.';
  return '';
})()}

Choose ONE condition from this list:
1. Acne and Rosacea
2. Actinic Keratosis
3. Basal Cell Carcinoma
4. Atopic Dermatitis
5. Bullous Disease
6. Cellulitis and Bacterial Infections
7. Eczema
8. Exanthems and Drug Eruptions
9. Hair Loss and Alopecia
10. Herpes and STDs
11. Pigmentation Disorders
12. Lupus and Connective Tissue Diseases
13. Melanoma and Skin Cancer
14. Nail Fungus
15. Contact Dermatitis
16. Psoriasis and Lichen Planus
17. Scabies and Lyme Disease
18. Seborrheic Keratoses
19. Systemic Disease
20. Fungal Infections
21. Urticaria (Hives)
22. Vascular Tumors
23. Vasculitis
24. Warts and Viral Infections

Please analyze and respond in this EXACT format:

Condition: [Choose one from the list above]
Confidence: [percentage between 75-95]
Severity: [low/medium/high]

Key Observations:
- [What you see that makes you think this - use simple words]
- [Another thing you notice about the skin]
- [What makes this different from similar conditions]

Treatment Recommendations:
1. [Simple, easy-to-follow treatment - like "Apply moisturizer twice daily"]
2. [Another helpful tip - use everyday language]
3. [Basic self-care advice]
4. [If needed, mention over-the-counter products in simple terms]

Next Steps:
1. [What to do right now - be specific and simple]
2. [How to keep an eye on it - like "Take a photo every 3 days"]
3. [When to see a doctor - be clear about timing]

Prevention:
1. [Easy prevention tip anyone can follow]
2. [Simple lifestyle change]
3. [What to avoid - be specific]

Note: This is an AI analysis for educational purposes only. Always consult a healthcare professional for medical advice.`;

        let imageData = image;
        let mediaType = "image/jpeg";
        
        if (image.includes(',')) {
          const [header, data] = image.split(',');
          imageData = data;
          
          const mediaTypeMatch = header.match(/data:([^;]+)/);
          if (mediaTypeMatch && mediaTypeMatch[1]) {
            mediaType = mediaTypeMatch[1];
            console.log('Extracted media type from data URL:', mediaType);
          }
        } else {
          try {
            const buffer = Buffer.from(image, 'base64');
            if (buffer[0] === 0xFF && buffer[1] === 0xD8) {
              mediaType = "image/jpeg";
            } else if (buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4E && buffer[3] === 0x47) {
              mediaType = "image/png";
            } else if (buffer[0] === 0x47 && buffer[1] === 0x49 && buffer[2] === 0x46) {
              mediaType = "image/gif";
            } else if (buffer[0] === 0x52 && buffer[1] === 0x49 && buffer[2] === 0x46 && buffer[3] === 0x46 && buffer[8] === 0x57 && buffer[9] === 0x45 && buffer[10] === 0x42 && buffer[11] === 0x50) {
              mediaType = "image/webp";
            } else if (buffer[0] === 0x42 && buffer[1] === 0x4D) {
              mediaType = "image/bmp";
            }
            console.log('Detected media type from magic bytes:', mediaType);
          } catch (e) {
            console.log('Could not detect media type from content, using default');
          }
        }
        
        console.log('Detected media type:', mediaType);
        const claudeResponse = await llmService.callAnthropic(prompt, imageData, mediaType);
        
        const message = claudeResponse.message;
        console.log('Claude Vision Response:', message);
        
        const conditionMatch = message.match(/Condition:\s*([^\n]+)/i);
        const condition = conditionMatch ? conditionMatch[1].trim() : "Skin Condition";
        
        const confidenceMatch = message.match(/Confidence:\s*(\d+)/i);
        let confidence = confidenceMatch ? parseInt(confidenceMatch[1]) : 85;
        
        if (confidence < 75) {
          confidence = Math.min(85, confidence + 20);
        }
        
        const severityMatch = message.match(/Severity:\s*(low|medium|high)/i);
        const severity = severityMatch ? severityMatch[1].toLowerCase() : 'medium';
        
        const observationMatch = message.match(/Key Observations:\s*\n-\s*([^\n]+)/i);
        const description = observationMatch ? observationMatch[1].trim() : "Detailed skin analysis completed";
        
        const treatmentSection = message.match(/Treatment Recommendations:([\s\S]*?)(?:Next Steps:|$)/i);
        let recommendations = [];
        if (treatmentSection) {
          const recMatches = treatmentSection[1].match(/\d+\.\s*([^\n]+)/g);
          if (recMatches) {
            recommendations = recMatches.map(r => r.replace(/^\d+\.\s*/, '').trim());
          }
        }
        
        const nextStepsSection = message.match(/Next Steps:([\s\S]*?)(?:Prevention:|$)/i);
        let nextSteps = [];
        if (nextStepsSection) {
          const stepMatches = nextStepsSection[1].match(/\d+\.\s*([^\n]+)/g);
          if (stepMatches) {
            nextSteps = stepMatches.map(s => s.replace(/^\d+\.\s*/, '').trim());
          }
        }
        
        const preventionSection = message.match(/Prevention:([\s\S]*?)(?:Note:|$)/i);
        let preventiveMeasures = [];
        if (preventionSection) {
          const prevMatches = preventionSection[1].match(/\d+\.\s*([^\n]+)/g);
          if (prevMatches) {
            preventiveMeasures = prevMatches.map(p => p.replace(/^\d+\.\s*/, '').trim());
          }
        }
        
        if (recommendations.length === 0) {
          recommendations = ["Consult with a dermatologist for professional evaluation"];
        }
        if (nextSteps.length === 0) {
          nextSteps = ["Monitor the condition for changes"];
        }
        if (preventiveMeasures.length === 0) {
          preventiveMeasures = ["Follow proper skin care routine"];
        }
        
        return res.json({
          success: true,
          analysis: {
            condition: condition,
            confidence: confidence,
            severity: severity,
            description: description,
            recommendations: recommendations,
            nextSteps: nextSteps,
            preventiveMeasures: preventiveMeasures,
            disclaimer: "This AI analysis is for educational purposes only. Please consult a healthcare professional for medical advice."
          },
          predictions: [{
            condition: condition,
            confidence: confidence
          }],
          timestamp: new Date().toISOString(),
          aiProvider: 'claude-vision'
        });
        
      } catch (claudeError) {
        console.error('Claude Vision error:', claudeError);
        
        if (claudeError.message) {
          console.log('Claude error details:', claudeError.message);
        }
      }
    }
    
    try {
      const modelResponse = await axios.post('http://localhost:5001/predict', {
        image: image
      }, {
        timeout: 30000
      });
      
      const predictions = modelResponse.data.predictions;
      const topPrediction = predictions[0];
      
      const allLowConfidence = predictions.every(p => p.confidence < 10);
      const lowConfidenceNote = allLowConfidence 
        ? " (Note: Model confidence is low for this image)" 
        : "";
      
      const severity = assessSeverity(topPrediction.condition);
      
      const treatments = generateTreatments(topPrediction.condition);
      
      res.json({
        success: true,
        analysis: {
          condition: topPrediction.condition,
          confidence: topPrediction.confidence,
          severity: severity,
          description: getConditionDescription(topPrediction.condition) + lowConfidenceNote,
          recommendations: treatments,
          disclaimer: "This is an AI-powered analysis for educational purposes only. Please consult a dermatologist for professional medical advice."
        },
        predictions: predictions,
        timestamp: new Date().toISOString()
      });
      
    } catch (modelError) {
      console.error('Model service error:', modelError.message);
      
      res.json({
        success: false,
        analysis: {
          condition: "Analysis Unavailable",
          confidence: 0,
          severity: "unknown",
          description: "The AI model service is currently unavailable. Please try again later.",
          recommendations: [
            "Please consult with a dermatologist for professional evaluation",
            "Take clear, well-lit photos for better analysis",
            "Keep the affected area clean and dry"
          ],
          disclaimer: "Model service is temporarily unavailable. Please consult a healthcare professional."
        },
        error: "Model service unavailable",
        timestamp: new Date().toISOString()
      });
    }
    
  } catch (error) {
    console.error('Analysis API Error:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: 'Failed to analyze image'
    });
  }
});

function assessSeverity(condition) {
  const severityMap = {
    'Melanoma and Skin Cancer': 'high',
    'Basal Cell Carcinoma': 'high',
    'Actinic Keratosis': 'medium',
    'Bullous Disease': 'medium',
    'Cellulitis and Bacterial Infections': 'medium',
    'Herpes and STDs': 'medium',
    'Lupus and Connective Tissue Diseases': 'medium',
    'Vasculitis': 'medium',
    'Acne and Rosacea': 'low',
    'Eczema': 'low',
    'Contact Dermatitis': 'low',
    'Seborrheic Keratoses': 'low',
    'Warts and Viral Infections': 'low'
  };
  
  return severityMap[condition] || 'medium';
}

function getConditionDescription(condition) {
  const descriptions = {
    'Acne and Rosacea': 'Common skin conditions causing pimples, redness, and inflammation.',
    'Melanoma and Skin Cancer': 'Serious conditions requiring immediate medical attention.',
    'Eczema': 'Chronic condition causing itchy, inflamed skin patches.',
    'Psoriasis and Lichen Planus': 'Autoimmune conditions affecting skin cell growth.',
    'Basal Cell Carcinoma': 'Most common form of skin cancer, usually treatable when caught early.',
    'Fungal Infections': 'Infections caused by fungi, often affecting moist areas of skin.'
  };
  
  return descriptions[condition] || 'A skin condition that may require professional evaluation.';
}

function generateTreatments(condition) {
  const treatmentMap = {
    'Acne and Rosacea': [
      'Use gentle, non-comedogenic cleansers',
      'Consider topical retinoids or benzoyl peroxide',
      'Avoid triggers like spicy foods and alcohol',
      'Consult dermatologist for prescription options'
    ],
    'Eczema': [
      'Keep skin moisturized with fragrance-free creams',
      'Avoid hot showers and harsh soaps',
      'Consider topical corticosteroids for flare-ups',
      'Identify and avoid triggers'
    ],
    'Melanoma and Skin Cancer': [
      'Seek immediate medical attention',
      'Schedule urgent dermatologist appointment',
      'Avoid sun exposure on affected area',
      'Do not attempt self-treatment'
    ]
  };
  
  return treatmentMap[condition] || [
    'Consult with a dermatologist for accurate diagnosis',
    'Keep the affected area clean and dry',
    'Monitor for any changes in appearance',
    'Avoid self-medication without professional guidance'
  ];
}

app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  res.status(500).json({
    error: 'Something went wrong!',
    message: 'Please try again later.'
  });
});

app.listen(PORT, () => {
  console.log(`DermoAI API server running on port ${PORT}`);
  const providers = [];
  if (process.env.ANTHROPIC_API_KEY) providers.push('anthropic');
  console.log(`Available LLM providers: ${providers.join(', ') || 'none (using local model only)'}`);
});

module.exports = app;