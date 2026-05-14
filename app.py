# app.py - Satellite Image Classifier with Dark Theme (FIXED)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import uvicorn

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "ResNet50_finetuned_best.pth"
CLASS_NAMES = ['🌫️ Cloudy', '🏜️ Desert', '🌿 Green Area', '💧 Water']
CLASS_KEYS = ['cloudy', 'desert', 'green_area', 'water']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# LOAD AI MODEL
# ============================================
print("🛰️  Initializing AI Satellite Classifier...")
print("🤖 Loading ResNet50 neural network...")

# Create model architecture
model = models.resnet50(weights=None)  # Fixed: use 'weights' instead of 'pretrained'
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 4)
)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Fixed: Convert device to string for display
device_str = str(DEVICE).upper()
print(f"✅ AI Model Ready on {device_str}")
print(f"📡 Monitoring: {', '.join(CLASS_NAMES)}")

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(
    title="🛰️ AI Satellite Vision",
    description="Deep Learning Model for Satellite Image Classification",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "model": "ResNet50",
        "accuracy": "99.88%",
        "f1_score": "0.9987",
        "device": device_str,
        "classes": CLASS_KEYS
    }

@app.get("/api/classes")
async def get_classes():
    return {
        "classes": [
            {"name": "cloudy", "icon": "🌫️", "description": "Cloud-covered regions"},
            {"name": "desert", "icon": "🏜️", "description": "Arid sandy terrain"},
            {"name": "green_area", "icon": "🌿", "description": "Vegetation & forests"},
            {"name": "water", "icon": "💧", "description": "Oceans, lakes & rivers"}
        ]
    }

@app.post("/api/predict")
async def predict_satellite(file: UploadFile = File(...)):
    """AI-powered satellite image classification"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Invalid file format. Please upload an image.")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # AI Prediction
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        # Prepare response
        all_probs = {
            CLASS_KEYS[i]: round(probabilities[0][i].item(), 4)
            for i in range(4)
        }
        
        return {
            "prediction": CLASS_KEYS[predicted_idx],
            "prediction_display": CLASS_NAMES[predicted_idx],
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "probabilities": all_probs,
            "model_used": "ResNet50"
        }
    
    except Exception as e:
        raise HTTPException(500, f"AI inference failed: {str(e)}")

# ============================================
# DARK THEME WEB INTERFACE
# ============================================

@app.get("/", response_class=HTMLResponse)
async def dark_theme_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Satellite Vision | Neural Earth Observation</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1428 100%);
                color: #e0e0e0;
                overflow-x: hidden;
            }
            
            /* Animated background stars */
            .stars {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 0;
            }
            
            .star {
                position: absolute;
                background: white;
                border-radius: 50%;
                opacity: 0;
                animation: twinkle 3s infinite;
            }
            
            @keyframes twinkle {
                0%, 100% { opacity: 0; }
                50% { opacity: 1; }
            }
            
            /* Orbiting satellite */
            .satellite {
                position: fixed;
                top: 20%;
                right: -100px;
                width: 80px;
                height: 80px;
                font-size: 60px;
                animation: orbit 20s linear infinite;
                pointer-events: none;
                z-index: 1;
            }
            
            @keyframes orbit {
                0% { transform: translateX(0) rotate(0deg); }
                100% { transform: translateX(-120vw) rotate(360deg); }
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 2rem;
                position: relative;
                z-index: 2;
            }
            
            /* Header */
            .header {
                text-align: center;
                margin-bottom: 3rem;
            }
            
            .ai-badge {
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 0.5rem 1.5rem;
                border-radius: 50px;
                font-size: 0.85rem;
                font-weight: 600;
                margin-bottom: 1rem;
                letter-spacing: 2px;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            
            h1 {
                font-size: 3.5rem;
                font-weight: 800;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 1rem;
            }
            
            .subtitle {
                font-size: 1.2rem;
                color: #a0a0c0;
                max-width: 600px;
                margin: 0 auto;
            }
            
            /* Stats Cards */
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.5rem;
                margin-bottom: 3rem;
            }
            
            .stat-card {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 1.5rem;
                text-align: center;
                transition: transform 0.3s, box-shadow 0.3s;
            }
            
            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
                border-color: rgba(102, 126, 234, 0.5);
            }
            
            .stat-icon {
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
            }
            
            .stat-value {
                font-size: 2rem;
                font-weight: 700;
                color: #667eea;
            }
            
            .stat-label {
                font-size: 0.85rem;
                color: #a0a0c0;
                margin-top: 0.5rem;
            }
            
            /* Gallery Section */
            .gallery {
                margin-bottom: 3rem;
            }
            
            .section-title {
                font-size: 1.8rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                text-align: center;
                background: linear-gradient(135deg, #fff 0%, #a0a0c0 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .class-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
            }
            
            .class-card {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                overflow: hidden;
                transition: all 0.3s;
                cursor: pointer;
            }
            
            .class-card:hover {
                transform: scale(1.05);
                box-shadow: 0 15px 40px rgba(0,0,0,0.3);
                border-color: #667eea;
            }
            
            .class-image {
                width: 100%;
                height: 200px;
                object-fit: cover;
            }
            
            .class-info {
                padding: 1rem;
                text-align: center;
            }
            
            .class-name {
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            
            .class-desc {
                font-size: 0.85rem;
                color: #a0a0c0;
            }
            
            /* Upload Area */
            .upload-section {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border: 2px dashed rgba(102, 126, 234, 0.5);
                border-radius: 20px;
                padding: 3rem;
                text-align: center;
                margin-bottom: 2rem;
                transition: all 0.3s;
            }
            
            .upload-section:hover {
                border-color: #667eea;
                background: rgba(102, 126, 234, 0.1);
            }
            
            .upload-icon {
                font-size: 4rem;
                margin-bottom: 1rem;
            }
            
            .upload-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 1rem 2rem;
                font-size: 1.1rem;
                font-weight: 600;
                border-radius: 50px;
                cursor: pointer;
                transition: transform 0.3s;
                margin: 1rem 0;
            }
            
            .upload-btn:hover {
                transform: scale(1.05);
            }
            
            .file-input {
                display: none;
            }
            
            /* Results Section */
            .results-section {
                background: rgba(0, 0, 0, 0.5);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 2rem;
                display: none;
                animation: fadeIn 0.5s;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .prediction-header {
                text-align: center;
                margin-bottom: 2rem;
            }
            
            .prediction-badge {
                display: inline-block;
                background: linear-gradient(135deg, #667eea, #764ba2);
                padding: 0.5rem 2rem;
                border-radius: 50px;
                font-size: 1.5rem;
                font-weight: 700;
            }
            
            .confidence-circle {
                width: 200px;
                height: 200px;
                margin: 0 auto 2rem;
                position: relative;
            }
            
            canvas {
                width: 100%;
                height: 100%;
            }
            
            .confidence-text {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                text-align: center;
            }
            
            .confidence-percent {
                font-size: 2.5rem;
                font-weight: 800;
                color: #667eea;
            }
            
            .prob-bar {
                background: rgba(255,255,255,0.1);
                height: 40px;
                margin: 10px 0;
                border-radius: 10px;
                overflow: hidden;
                position: relative;
            }
            
            .prob-fill {
                height: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                display: flex;
                align-items: center;
                padding-left: 15px;
                color: white;
                font-weight: 600;
                transition: width 1s ease;
            }
            
            /* Loading Animation */
            .loader {
                display: none;
                text-align: center;
                margin: 2rem 0;
            }
            
            .loader-spinner {
                width: 50px;
                height: 50px;
                border: 3px solid rgba(102, 126, 234, 0.3);
                border-top-color: #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 1rem;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            /* Footer */
            .footer {
                text-align: center;
                padding: 2rem;
                margin-top: 3rem;
                border-top: 1px solid rgba(255,255,255,0.1);
                color: #a0a0c0;
                font-size: 0.85rem;
            }
            
            @media (max-width: 768px) {
                h1 { font-size: 2rem; }
                .container { padding: 1rem; }
                .stats-grid { grid-template-columns: 1fr 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="stars" id="stars"></div>
        <div class="satellite">🛰️</div>
        
        <div class="container">
            <div class="header">
                <div class="ai-badge">
                    🤖 AI-POWERED | DEEP LEARNING | RESNET50
                </div>
                <h1>🛰️ AI Satellite Vision</h1>
                <div class="subtitle">
                    Neural network-powered satellite image classification with 99.88% accuracy
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-icon">🎯</div>
                    <div class="stat-value">99.88%</div>
                    <div class="stat-label">Classification Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">⚡</div>
                    <div class="stat-value">&lt;50ms</div>
                    <div class="stat-label">Inference Time</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">📊</div>
                    <div class="stat-value">4</div>
                    <div class="stat-label">Land Cover Classes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">🧠</div>
                    <div class="stat-value">25M</div>
                    <div class="stat-label">Neural Parameters</div>
                </div>
            </div>
            
            <div class="gallery">
                <h2 class="section-title">📡 Earth Observation Classes</h2>
                <div class="class-grid" id="classGrid"></div>
            </div>
            
            <div class="upload-section">
                <div class="upload-icon">📸</div>
                <h3>Upload Satellite Imagery</h3>
                <p style="color: #a0a0c0; margin: 1rem 0;">Drag & drop or click to select</p>
                <input type="file" id="fileInput" class="file-input" accept="image/*">
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                    🚀 Select Image
                </button>
                <p style="font-size: 0.8rem; margin-top: 1rem;">Supported: JPG, PNG, JPEG</p>
            </div>
            
            <div class="loader" id="loader">
                <div class="loader-spinner"></div>
                <p>🤖 AI analyzing satellite imagery...</p>
            </div>
            
            <div class="results-section" id="results">
                <div class="prediction-header">
                    <div class="prediction-badge" id="predictionBadge">—</div>
                </div>
                
                <div class="confidence-circle">
                    <canvas id="confidenceCanvas" width="200" height="200"></canvas>
                    <div class="confidence-text">
                        <div class="confidence-percent" id="confidencePercent">0%</div>
                        <div style="font-size: 0.8rem;">Confidence</div>
                    </div>
                </div>
                
                <div id="probabilities"></div>
            </div>
            
            <div class="footer">
                <p>🚀 Powered by ResNet50 Deep Neural Network | Trained on 1,200+ Satellite Images</p>
                <p style="margin-top: 0.5rem;">🌍 Real-time Earth Observation | 99.88% Test Accuracy | F1-Score: 0.9987</p>
            </div>
        </div>
        
        <script>
            // Generate stars
            for(let i = 0; i < 100; i++) {
                const star = document.createElement('div');
                star.className = 'star';
                star.style.left = Math.random() * 100 + '%';
                star.style.top = Math.random() * 100 + '%';
                star.style.width = Math.random() * 3 + 'px';
                star.style.height = star.style.width;
                star.style.animationDelay = Math.random() * 3 + 's';
                document.getElementById('stars').appendChild(star);
            }
            
            // Class data
            const classes = [
                { name: 'Cloudy', key: 'cloudy', icon: '🌫️', desc: 'Atmospheric cloud formations', color: '#8899aa', img: 'https://images.unsplash.com/photo-1483728642387-6c3bdd6c93e5?w=400' },
                { name: 'Desert', key: 'desert', icon: '🏜️', desc: 'Arid sandy terrain & dunes', color: '#e8c48a', img: 'https://images.unsplash.com/photo-1509316785289-025f5b846b35?w=400' },
                { name: 'Green Area', key: 'green_area', icon: '🌿', desc: 'Forests, crops & vegetation', color: '#5cad5c', img: 'https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=400' },
                { name: 'Water', key: 'water', icon: '💧', desc: 'Oceans, lakes & rivers', color: '#4a90d9', img: 'https://images.unsplash.com/photo-1437622368342-7a3d73a34c8f?w=400' }
            ];
            
            // Render class gallery
            const classGrid = document.getElementById('classGrid');
            classes.forEach(c => {
                const card = document.createElement('div');
                card.className = 'class-card';
                card.innerHTML = `
                    <img src="${c.img}" class="class-image" alt="${c.name}">
                    <div class="class-info">
                        <div class="class-name">${c.icon} ${c.name}</div>
                        <div class="class-desc">${c.desc}</div>
                    </div>
                `;
                classGrid.appendChild(card);
            });
            
            // File upload handler
            document.getElementById('fileInput').addEventListener('change', async (e) => {
                const file = e.target.files[0];
                if (!file) return;
                
                // Show loader
                document.getElementById('loader').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                
                // Upload and predict
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    displayResults(result);
                } catch (error) {
                    console.error('Error:', error);
                    alert('AI analysis failed. Please try again.');
                } finally {
                    document.getElementById('loader').style.display = 'none';
                }
            });
            
            function displayResults(result) {
                const prediction = result.prediction_display;
                const confidence = result.confidence_percent;
                
                // Update badge
                document.getElementById('predictionBadge').innerHTML = prediction;
                
                // Update confidence
                document.getElementById('confidencePercent').innerHTML = confidence + '%';
                
                // Draw confidence circle
                drawConfidenceCircle(confidence / 100);
                
                // Render probabilities
                const probsDiv = document.getElementById('probabilities');
                probsDiv.innerHTML = '<h3>📊 Neural Network Activations</h3>';
                
                const classMap = {
                    'cloudy': '🌫️ Cloudy',
                    'desert': '🏜️ Desert', 
                    'green_area': '🌿 Green Area',
                    'water': '💧 Water'
                };
                
                for (const [key, prob] of Object.entries(result.probabilities)) {
                    const percent = prob * 100;
                    probsDiv.innerHTML += `
                        <div class="prob-bar">
                            <div class="prob-fill" style="width: ${percent}%">
                                ${classMap[key]} - ${percent.toFixed(1)}%
                            </div>
                        </div>
                    `;
                }
                
                document.getElementById('results').style.display = 'block';
            }
            
            function drawConfidenceCircle(percent) {
                const canvas = document.getElementById('confidenceCanvas');
                const ctx = canvas.getContext('2d');
                const centerX = canvas.width / 2;
                const centerY = canvas.height / 2;
                const radius = 80;
                
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Background circle
                ctx.beginPath();
                ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
                ctx.strokeStyle = 'rgba(255,255,255,0.2)';
                ctx.lineWidth = 15;
                ctx.stroke();
                
                // Foreground circle (confidence)
                const startAngle = -Math.PI / 2;
                const endAngle = startAngle + (2 * Math.PI * percent);
                
                ctx.beginPath();
                ctx.arc(centerX, centerY, radius, startAngle, endAngle);
                ctx.strokeStyle = '#667eea';
                ctx.lineWidth = 15;
                ctx.stroke();
                
                // Glow effect
                ctx.shadowBlur = 20;
                ctx.shadowColor = '#667eea';
                ctx.stroke();
                ctx.shadowBlur = 0;
            }
            
            // Drag and drop
            const dropZone = document.querySelector('.upload-section');
            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.style.borderColor = '#667eea';
                dropZone.style.background = 'rgba(102, 126, 234, 0.1)';
            });
            
            dropZone.addEventListener('dragleave', () => {
                dropZone.style.borderColor = 'rgba(102, 126, 234, 0.5)';
                dropZone.style.background = 'rgba(255, 255, 255, 0.05)';
            });
            
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.style.borderColor = 'rgba(102, 126, 234, 0.5)';
                dropZone.style.background = 'rgba(255, 255, 255, 0.05)';
                
                const file = e.dataTransfer.files[0];
                if (file && file.type.startsWith('image/')) {
                    const input = document.getElementById('fileInput');
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    input.files = dataTransfer.files;
                    input.dispatchEvent(new Event('change'));
                }
            });
        </script>
    </body>
    </html>
    """

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🛰️  AI SATELLITE VISION - READY FOR LAUNCH               ║
    ║                                                              ║
    ║   🤖 Model: ResNet50 (99.88% Accuracy)                      ║
    ║   🌍 Classes: Cloudy | Desert | Green Area | Water          ║
    ║                                                              ║
    ║   🚀 Server running at: http://localhost:8000               ║
    ║   📚 API Docs: http://localhost:8000/api/docs               ║
    ║   🎨 Web UI: http://localhost:8000                          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )