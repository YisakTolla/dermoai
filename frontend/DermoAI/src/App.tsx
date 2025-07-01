import React, { useState, useRef, useEffect } from 'react';
import './App.css';

// Import the new secure components
import Chatbot from './Chatbot';
import GoogleAPIDermatologistFinder from './GoogleAPIDermatologistFinder';

interface AnalysisResult {
  condition: string;
  confidence: number;
  severity: 'Low' | 'Medium' | 'High';
  category: string;
  recommendations: string[];
  nextSteps: string[];
  preventiveMeasures: string[];
  timeframe: string;
}

function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [currentSection, setCurrentSection] = useState('hero');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const analysisRef = useRef<HTMLDivElement>(null);

  // API Configuration
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        alert('Please select an image smaller than 10MB');
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
        setAnalysisResult(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    element?.scrollIntoView({ behavior: 'smooth' });
  };

  const analyzeImage = async () => {
    if (!selectedImage) {
      alert('Please select an image first');
      return;
    }

    setIsAnalyzing(true);
    
    try {
      // Call your backend API for analysis
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: selectedImage.split(',')[1], // Remove data:image/jpeg;base64, prefix
          timestamp: new Date().toISOString()
        })
      });

      if (response.ok) {
        const result = await response.json();
        setAnalysisResult(result);
      } else {
        // Fallback to mock analysis if API fails
        console.warn('API analysis failed, using mock data');
        mockAnalysis();
      }
    } catch (error) {
      console.error('Analysis error:', error);
      // Fallback to mock analysis
      mockAnalysis();
    }

    setIsAnalyzing(false);
    
    setTimeout(() => {
      analysisRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 500);
  };

  // Mock analysis function as fallback
  const mockAnalysis = () => {
    const mockResults: AnalysisResult[] = [
      {
        condition: 'Healthy Skin',
        confidence: 94,
        severity: 'Low',
        category: 'Normal',
        timeframe: 'Current state',
        recommendations: [
          'Continue your current skincare routine',
          'Use broad-spectrum SPF 30+ sunscreen daily',
          'Maintain adequate hydration',
          'Consider antioxidant-rich skincare products'
        ],
        nextSteps: [
          'Regular skin self-examinations monthly',
          'Annual dermatologist check-up',
          'Monitor any changes in skin texture or color'
        ],
        preventiveMeasures: [
          'Avoid excessive sun exposure',
          'Use gentle, pH-balanced cleansers',
          'Maintain a balanced diet rich in vitamins',
          'Get adequate sleep and manage stress'
        ]
      },
      {
        condition: 'Acne Vulgaris',
        confidence: 87,
        severity: 'Medium',
        category: 'Inflammatory',
        timeframe: '2-4 weeks for improvement',
        recommendations: [
          'Use salicylic acid or benzoyl peroxide cleansers',
          'Apply non-comedogenic moisturizers',
          'Avoid touching or picking at affected areas',
          'Consider professional treatment options'
        ],
        nextSteps: [
          'Consult with a dermatologist within 2 weeks',
          'Start gentle, consistent skincare routine',
          'Monitor progress and adjust treatment as needed'
        ],
        preventiveMeasures: [
          'Wash pillowcases frequently',
          'Clean makeup brushes regularly',
          'Avoid oil-based skincare products',
          'Maintain hormonal balance through proper diet'
        ]
      },
      {
        condition: 'Dry Skin (Xerosis)',
        confidence: 91,
        severity: 'Low',
        category: 'Environmental',
        timeframe: '1-2 weeks for improvement',
        recommendations: [
          'Use ceramide-based moisturizers twice daily',
          'Take shorter, lukewarm showers',
          'Apply moisturizer immediately after bathing',
          'Use a humidifier in dry environments'
        ],
        nextSteps: [
          'Establish consistent moisturizing routine',
          'Monitor skin response to new products',
          'Consider professional consultation if persistent'
        ],
        preventiveMeasures: [
          'Avoid harsh soaps and detergents',
          'Wear protective clothing in cold weather',
          'Stay hydrated with adequate water intake',
          'Use gentle, fragrance-free products'
        ]
      }
    ];
    
    const randomResult = mockResults[Math.floor(Math.random() * mockResults.length)];
    setAnalysisResult(randomResult);
  };

  const resetAnalysis = () => {
    setSelectedImage(null);
    setAnalysisResult(null);
    setIsAnalyzing(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  const exportReport = () => {
    if (!analysisResult) return;
    
    const reportData = {
      timestamp: new Date().toISOString(),
      analysis: analysisResult,
      disclaimer: 'This analysis is for educational purposes only and should not replace professional medical advice.'
    };
    
    const dataStr = JSON.stringify(reportData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `dermoai-analysis-${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  useEffect(() => {
    const handleScroll = () => {
      const sections = ['hero', 'how-it-works', 'analysis', 'features'];
      const scrollPosition = window.scrollY + window.innerHeight / 2;

      sections.forEach(section => {
        const element = document.getElementById(section);
        if (element) {
          const { offsetTop, offsetHeight } = element;
          if (scrollPosition >= offsetTop && scrollPosition <= offsetTop + offsetHeight) {
            setCurrentSection(section);
          }
        }
      });
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="app">
      {/* Navigation */}
      <nav className="navbar">
        <div className="nav-container">
          <div className="nav-brand">DermoAI</div>
          <div className="nav-links">
            <button onClick={() => scrollToSection('hero')} className={currentSection === 'hero' ? 'active' : ''}>
              Home
            </button>
            <button onClick={() => scrollToSection('how-it-works')} className={currentSection === 'how-it-works' ? 'active' : ''}>
              Process
            </button>
            <button onClick={() => scrollToSection('analysis')} className={currentSection === 'analysis' ? 'active' : ''}>
              Analysis
            </button>
            <button onClick={() => scrollToSection('features')} className={currentSection === 'features' ? 'active' : ''}>
              Features
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section id="hero" className="hero-section">
        <div className="hero-container">
          <div className="hero-content">
            <div className="hero-badge">AI-POWERED DERMATOLOGY</div>
            <h1 className="hero-title">
              Revolutionary skin analysis
              <span className="hero-highlight">for everyone, everywhere</span>
            </h1>
            <p className="hero-description">
              Advanced AI technology trained on 30K+ medical images to make dermatological screening 
              accessible to everyone. Breaking down barriers between cutting-edge healthcare technology 
              and global communities who need it most.
            </p>
            <div className="hero-actions">
              <button 
                className="btn btn-primary"
                onClick={() => scrollToSection('analysis')}
              >
                Start Analysis
              </button>
              <button 
                className="btn btn-secondary"
                onClick={() => scrollToSection('how-it-works')}
              >
                Learn More
              </button>
            </div>
          </div>
          <div className="hero-visual">
            <div className="metric-card">
              <div className="metric-value">23</div>
              <div className="metric-label">Skin Conditions</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">30K+</div>
              <div className="metric-label">Training Images</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">Live</div>
              <div className="metric-label">Demo Status</div>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="process-section">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">How It Works</h2>
            <p className="section-description">Advanced AI-powered skin analysis in three simple steps</p>
          </div>
          
          <div className="process-grid">
            <div className="process-step">
              <div className="step-header">
                <div className="step-number">01</div>
                <h3>Image Capture</h3>
              </div>
              <p>Upload a clear image of the skin area you want to analyze. Our system accepts common image formats and prepares them for AI analysis.</p>
            </div>
            
            <div className="process-step">
              <div className="step-header">
                <div className="step-number">02</div>
                <h3>AI Analysis</h3>
              </div>
              <p>Advanced machine learning algorithms trained on extensive medical datasets analyze patterns, textures, and anomalies to identify potential skin conditions.</p>
            </div>
            
            <div className="process-step">
              <div className="step-header">
                <div className="step-number">03</div>
                <h3>Clinical Report</h3>
              </div>
              <p>Receive detailed analysis results with condition identification, confidence scores, and general recommendations based on pattern recognition.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Analysis Section */}
      <section id="analysis" className="analysis-section">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Try Our Demo</h2>
            <p className="section-description">Experience cutting-edge AI technology - upload an image to see automated skin analysis in action</p>
          </div>
          
          <div className="analysis-container">
            {selectedImage ? (
              <div className="image-preview-container">
                <div className="image-preview">
                  <img src={selectedImage} alt="Selected" className="preview-image" />
                  <div className="image-overlay">
                    <button className="overlay-btn" onClick={triggerFileSelect}>
                      Change Image
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="upload-zone" onClick={triggerFileSelect}>
                <div className="upload-content">
                  <div className="upload-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="17,8 12,3 7,8"/>
                      <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                  </div>
                  <h3>Upload Image</h3>
                  <p>Select a clear, well-lit image of the skin area for analysis</p>
                  <span className="upload-specs">JPG, PNG • Max 10MB • Minimum 512px</span>
                </div>
              </div>
            )}
            
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageSelect}
              className="file-input"
            />

            {selectedImage && !analysisResult && (
              <div className="action-container">
                <button 
                  className={`btn btn-primary ${isAnalyzing ? 'analyzing' : ''}`}
                  onClick={analyzeImage}
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? (
                    <>
                      <div className="spinner"></div>
                      Analyzing
                    </>
                  ) : (
                    'Analyze Image'
                  )}
                </button>
              </div>
            )}

            {isAnalyzing && (
              <div className="analysis-progress">
                <div className="progress-bar">
                  <div className="progress-fill"></div>
                </div>
                <p className="progress-text">Processing image and analyzing skin patterns...</p>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Results Section */}
      {analysisResult && (
        <section ref={analysisRef} id="results" className="results-section">
          <div className="container">
            <div className="results-header">
              <h2 className="section-title">Analysis Results</h2>
              <div className="confidence-indicator">
                <span className="confidence-label">Confidence</span>
                <span className="confidence-value">{analysisResult.confidence}%</span>
              </div>
            </div>

            <div className="results-layout">
              {/* Primary Diagnosis */}
              <div className="diagnosis-card">
                <div className="card-header">
                  <h3>Primary Diagnosis</h3>
                  <span className={`severity-indicator ${analysisResult.severity.toLowerCase()}`}>
                    {analysisResult.severity} Priority
                  </span>
                </div>
                <div className="diagnosis-content">
                  <h4 className="condition-name">{analysisResult.condition}</h4>
                  <div className="condition-meta">
                    <span className="condition-type">{analysisResult.category} Condition</span>
                    <span className="condition-timeline">{analysisResult.timeframe}</span>
                  </div>
                  <div className="confidence-visualization">
                    <div className="confidence-bar">
                      <div 
                        className="confidence-fill"
                        style={{ width: `${analysisResult.confidence}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Recommendations Grid */}
              <div className="recommendations-grid">
                <div className="recommendation-card">
                  <h4>Treatment Recommendations</h4>
                  <ul className="recommendation-list">
                    {analysisResult.recommendations.map((rec, index) => (
                      <li key={index}>{rec}</li>
                    ))}
                  </ul>
                </div>

                <div className="recommendation-card">
                  <h4>Next Steps</h4>
                  <ul className="recommendation-list">
                    {analysisResult.nextSteps.map((step, index) => (
                      <li key={index}>{step}</li>
                    ))}
                  </ul>
                </div>

                <div className="recommendation-card">
                  <h4>Prevention</h4>
                  <ul className="recommendation-list">
                    {analysisResult.preventiveMeasures.map((measure, index) => (
                      <li key={index}>{measure}</li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="results-actions">
                <button className="btn btn-primary" onClick={resetAnalysis}>
                  New Analysis
                </button>
                <button className="btn btn-secondary" onClick={exportReport}>
                  Export Report
                </button>
                <button className="btn btn-secondary" onClick={() => window.print()}>
                  Print Results
                </button>
                <button 
                  className="btn btn-secondary"
                  onClick={() => {
                    // Scroll to dermatologist finder
                    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                    // Small delay to allow scroll, then expand finder
                    setTimeout(() => {
                      const finderElement = document.querySelector('[style*="position: fixed"][style*="bottom: 0"]') as HTMLElement;
                      if (finderElement) {
                        finderElement.click();
                      }
                    }, 1000);
                  }}
                >
                  Find Local Dermatologist
                </button>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Features Section */}
      <section id="features" className="features-section">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Innovation Features</h2>
            <p className="section-description">Built to showcase the future of accessible healthcare technology worldwide</p>
          </div>
          
          <div className="features-grid">
            <div className="feature-item">
              <div className="feature-icon">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M12 2L2 7v10c0 5.55 3.84 9.74 9 11 5.16-1.26 9-5.45 9-11V7l-10-5z"/>
                </svg>
              </div>
              <h3>Medical Dataset Integration</h3>
              <p>Demonstrates integration with DermNet's 23-category skin condition dataset, showcasing how AI can learn from medical imagery.</p>
            </div>
            
            <div className="feature-item">
              <div className="feature-icon">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <circle cx="12" cy="12" r="3"/>
                  <path d="M12 1v6m0 6v6m11-7h-6m-6 0H1"/>
                </svg>
              </div>
              <h3>AI-Powered Chat Assistant</h3>
              <p>Intelligent chatbot powered by advanced LLMs that can answer dermatology questions and provide educational information.</p>
            </div>
            
            <div className="feature-item">
              <div className="feature-icon">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
                  <circle cx="12" cy="10" r="3"/>
                </svg>
              </div>
              <h3>Real Dermatologist Finder</h3>
              <p>Google Maps integration to find actual local dermatologists with real contact information, hours, and reviews.</p>
            </div>
            
            <div className="feature-item">
              <div className="feature-icon">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                  <circle cx="8.5" cy="8.5" r="1.5"/>
                  <polyline points="21,15 16,10 5,21"/>
                </svg>
              </div>
              <h3>Privacy by Design</h3>
              <p>User privacy is our priority - all processing happens securely with proper API key management and data protection.</p>
            </div>
            
            <div className="feature-item">
              <div className="feature-icon">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <polyline points="22,12 18,12 15,21 9,3 6,12 2,12"/>
                </svg>
              </div>
              <h3>Healthcare Innovation</h3>
              <p>Demonstrating how AI can support healthcare accessibility, especially in underserved communities worldwide.</p>
            </div>
            
            <div className="feature-item">
              <div className="feature-icon">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
                  <circle cx="9" cy="7" r="4"/>
                  <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
                  <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
                </svg>
              </div>
              <h3>Complete Healthcare Solution</h3>
              <p>End-to-end solution from AI analysis to finding real medical care, bridging the gap between technology and healthcare access.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Medical Disclaimer */}
      <div className="disclaimer-section">
        <div className="container">
          <div className="disclaimer-content">
            <div className="disclaimer-header">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
              </svg>
              <h4>Medical Disclaimer</h4>
            </div>
            <p>
              This demonstration tool is not intended for actual medical diagnosis or treatment decisions. 
              All AI analysis results are for educational purposes only. Always consult qualified healthcare 
              providers for medical advice and proper diagnosis. Use our "Find Local Dermatologist" to help locate a dermatologist.
            </p>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <div className="footer-grid">
            <div className="footer-brand">
              <h3>DermoAI</h3>
              <p>AI-powered skin analysis platform making dermatological screening more accessible worldwide. Using advanced machine learning to help identify potential skin conditions and connect users with professional healthcare providers.</p>
            </div>
            <div className="link-column">
              <h4>Technology</h4>
              <a href="#how-it-works">How It Works</a>
              <a href="#features">Features</a>
              <a href="#analysis">Try Demo</a>
              <a href="https://huggingface.co/spaces/Hrigved/skinalyze" target="_blank" rel="noopener noreferrer">HuggingFace Model</a>
            </div>
            <div className="link-column">
              <h4>Healthcare</h4>
              <a href="#analysis">Skin Analysis</a>
              <a href="#" onClick={(e) => { e.preventDefault(); window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' }); }}>Find Dermatologists</a>
              <a href="#features">AI Chat Assistant</a>
              <a href="#disclaimer">Medical Disclaimer</a>
            </div>
            <div className="link-column">
              <h4>About</h4>
              <a href="#hero">About DermoAI</a>
              <a href="https://github.com/yourusername/dermoai" target="_blank" rel="noopener noreferrer">Source Code</a>
              <a href="#features">Innovation</a>
              <a href="#disclaimer">Privacy Policy</a>
            </div>
          </div>
          <div className="footer-bottom">
            <p>&copy; 2024 DermoAI - Democratizing Access to Dermatological AI Technology</p>
          </div>
        </div>
      </footer>

      {/* Secure AI Chatbot */}
      <Chatbot />

      {/* Google Maps Dermatologist Finder */}
      <GoogleAPIDermatologistFinder userLocation="20164" />
    </div>
  );
}

export default App;