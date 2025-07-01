# DermoAI - Skin Disease Classification System

A professional web application for AI-powered skin disease classification using deep learning. The system provides real-time analysis of skin conditions with treatment recommendations and severity assessments.

## Overview

DermoAI uses a pre-trained deep learning model to classify 24 different skin conditions from uploaded images. The application features a modern React frontend, Express.js backend, and a Python-based ML service using PyTorch.

## Features

- **AI-Powered Analysis**: Classifies 24 different skin conditions using deep learning
- **Real-time Predictions**: Get instant analysis with confidence scores
- **Treatment Recommendations**: Provides condition-specific treatment suggestions
- **Severity Assessment**: Automatic severity classification (Low/Medium/High)
- **Dermatologist Finder**: Integrated Google Maps to find nearby specialists
- **AI Chatbot**: LLM-powered assistant for dermatology questions
- **Professional UI**: Clean, modern interface with responsive design

## Tech Stack

### Frontend
- React 19 with TypeScript
- Modern CSS with responsive design
- Google Maps API integration
- Real-time image preview and analysis

### Backend
- Node.js with Express.js
- Multi-LLM support (OpenAI, Google Gemini, Anthropic Claude)
- RESTful API architecture
- CORS-enabled with security middleware

### ML Service
- PyTorch (ResNet152/DenseNet121)
- Flask API for model serving
- 24-class skin disease classification
- Confidence scoring system

## Project Structure

```
skindiseasemlapp/
├── frontend/               # React frontend application
│   ├── public/            # Static assets
│   ├── src/               # Source code
│   │   ├── App.tsx        # Main application component
│   │   ├── Chatbot.tsx    # AI chatbot component
│   │   └── GoogleAPIDermatologistFinder.tsx
│   ├── package.json       # Frontend dependencies
│   └── tsconfig.json      # TypeScript configuration
│
├── backend/               # Node.js backend server
│   ├── model_service.py   # Python ML service
│   ├── server.js          # Express server
│   ├── routes/            # API routes
│   ├── classes.txt        # Disease classifications
│   ├── skin_disease_model.pth  # Model weights
│   ├── requirements.txt   # Python dependencies
│   └── package.json       # Node dependencies
│
└── README.md             # This file
```

## Prerequisites

- Node.js 16+ and npm
- Python 3.8+
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YisakTolla/dermoai.git
cd dermoai
```

### 2. Backend Setup

```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Install Node dependencies
npm install

# Configure environment variables
# Copy .env.example to .env and add your API keys
cp .env.example .env
```

### 3. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Configure environment variables
# Update .env with your configuration
```

## Configuration

### Backend Environment Variables (.env)

```env
# API Keys
ANTHROPIC_API_KEY=your-key-here
GOOGLE_MAPS_API_KEY=your-key-here

# Server
PORT=3001
FRONTEND_URL=http://localhost:3000
DEFAULT_LLM_PROVIDER=anthropic
```

### Frontend Environment Variables (.env)

```env
REACT_APP_API_URL=http://localhost:3001/api
REACT_APP_GOOGLE_MAPS_API_KEY=your-key-here
```

## Running the Application

### Start All Services

1. **ML Model Service** (Terminal 1):
```bash
cd backend
python model_service.py
```

2. **Backend Server** (Terminal 2):
```bash
cd backend
npm run dev
```

3. **Frontend** (Terminal 3):
```bash
cd frontend
npm start
```

The application will be available at `http://localhost:3000`

## API Endpoints

### POST /api/analyze
Analyzes uploaded skin images

**Request:**
```json
{
  "image": "base64_encoded_image"
}
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "condition": "Detected Condition",
    "confidence": 85.5,
    "severity": "medium",
    "recommendations": ["..."],
    "disclaimer": "..."
  }
}
```

### POST /api/chat
AI-powered dermatology assistant

### GET /api/health
Health check endpoint

## Supported Skin Conditions

The model can classify 24 different skin conditions including:
- Acne and Rosacea
- Melanoma and Skin Cancer
- Basal Cell Carcinoma
- Eczema
- Psoriasis
- Fungal Infections
- And 18 more conditions

## Development

### Code Style
- Frontend: TypeScript with React best practices
- Backend: ES6+ JavaScript with Express patterns
- Python: PEP 8 compliant

### Testing
```bash
# Frontend tests
cd frontend && npm test

# Backend tests
cd backend && npm test
```

## Deployment

### Quick Deploy (Free Hosting)

The application is designed for easy deployment on free platforms:

1. **Frontend**: Deploy to [Vercel](https://vercel.com)
   ```bash
   cd frontend && vercel --prod
   ```

2. **Backend**: Deploy to [Render](https://render.com)
   - Connect GitHub repo
   - Set root directory: `backend`
   - Add environment variables

3. **ML Model**: Already deployed at [Hugging Face Spaces](https://huggingface.co/spaces/Hrigved/skinalyze)

See `DEPLOYMENT_SIMPLE.md` for detailed instructions.

## Security Considerations

- All API keys are stored in environment variables
- CORS configured for production domains
- Rate limiting implemented on API endpoints
- Input validation and sanitization
- HTTPS required for production

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This application is for educational and demonstration purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## Acknowledgments

- Model architecture based on ResNet152/DenseNet121
- Dataset: DermNet skin disease dataset
- UI components: Lucide React icons
- LLM providers: OpenAI, Google, Anthropic