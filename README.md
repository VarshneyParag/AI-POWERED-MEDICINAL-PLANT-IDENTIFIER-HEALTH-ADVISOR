🌿 Plant Medic - AI-Powered Medicinal Plant Identifier & Health Advisor
Plant Medic is an innovative web application that combines the ancient wisdom of Ayurvedic medicine with modern artificial intelligence to identify medicinal plants and provide detailed health insights. This powerful tool can recognize 40+ Indian medicinal plants from images and deliver comprehensive information about their medicinal properties, Ayurvedic characteristics, health benefits, usage instructions, and precautions.

🎯 Key Features
🤖 AI-Powered Identification: Advanced Random Forest classifier with 90%+ accuracy trained on 180+ enhanced features

🌿 40+ Medicinal Plants Database: Comprehensive information on Indian medicinal plants including Aloe Vera, Tulsi, Neem, Ashwagandha, and many more

💊 Ayurvedic Insights: Detailed Ayurvedic properties (Rasa, Virya, Vipaka, Dosha) for each plant

🏥 Health Benefits & Uses: Complete medicinal uses, health benefits, and practical usage instructions

⚠️ Safety Precautions: Important warnings and usage guidelines for safe consumption

📱 Modern Web Interface: Beautiful, responsive design with drag-and-drop functionality

🔬 High Accuracy: Enhanced feature extraction with color, texture, shape, and histogram analysis

🌐 Real-time Analysis: Instant plant identification with confidence scoring

🛠️ Technology Stack
Backend: Python Flask with OpenCV, scikit-learn, NumPy

AI Model: Random Forest Classifier with 200 estimators

Frontend: HTML5, Tailwind CSS, JavaScript

Image Processing: OpenCV with advanced feature extraction

Data Storage: Joblib for model persistence

API: RESTful Flask API with CORS support

📊 Model Performance
Test Accuracy: 90%+ on medicinal plant dataset

Feature Extraction: 180+ features per image

Image Resolution: 512x512 pixels for optimal analysis

Confidence Scoring: Intelligent confidence levels with health assessments

Training Data: Comprehensive dataset of 40 medicinal plant species

🌱 Supported Plants
The system can identify 40 medicinal plants including Tulsi (Holy Basil), Neem, Aloe Vera, Ashwagandha, Brahmi, Amla, Turmeric, Ginger, Amruta Balli, Curry Leaf, Hibiscus, Jasmine, Lemon Grass, Mango, Mint, Pomegranate, Rose, and many more - each with complete Ayurvedic profiles and medicinal information.

📁 Project Structure

AI-POWERED MEDICINAL PLANT IDENTIFIER& HEALTH ADVISOR/
│
├── app.py                          # Main Flask application
├── train_model.py                  # Model training script
├── random_forest_model.pkl         # Trained model file
├── feature_scaler.pkl              # Feature scaler
├── class_mapping.pkl               # Class labels mapping
├── training_info.pkl               # Training metadata
│
├── templates/
│   └── index.html                  # Main web interface
│
├── uploads/                        # Temporary upload directory
│   └── (auto-created at runtime)
│
├── Medicinal plant dataset/        # Training dataset
│   ├── aloevera/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── tulsi/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── neem/
│   │   └── ...
│   └── ... (40 plant directories)
│
└── README.md                       # Project documentation

🚀 Installation & Setup
Prerequisites
Python 3.8+

pip package manager

💡 Usage
Upload Image: Drag & drop or click to upload a plant image

AI Analysis: The system automatically extracts features and identifies the plant

View Results: Get detailed information including:

Plant identification with confidence score

Ayurvedic properties (Rasa, Virya, Vipaka, Dosha)

Medicinal uses and health benefits

Usage instructions and precautions

Scientific classification

🔬 Technical Details
Feature Extraction
Color Features: BGR, HSV, LAB color spaces with statistical moments

Texture Features: Sobel gradients, Laplacian filters, Gabor filters, LBP, GLCM

Shape Features: Contour analysis, geometric properties

Histogram Features: Color distributions and entropy analysis

Model Architecture
Algorithm: Random Forest Classifier

Estimators: 200 trees

Feature Scaling: StandardScaler

Validation: 15% test split with stratification

API Endpoints
POST /predict - Plant identification from image

GET /model-info - Model metadata and status

GET /health - System health check

GET /medicinal-plants - List all supported plants

GET /plant-details/{plant_key} - Specific plant information

🌟 Unique Features
Ayurvedic Integration: Traditional Ayurvedic properties for each plant

Hindi Names: Native language support for Indian users

Safety First: Clear precautions and usage warnings

Confidence Scoring: Intelligent confidence levels for reliable results

Unknown Plant Detection: Handles unrecognized plants gracefully

Real-time Processing: Fast analysis with progress indicators

📝 Note
Plant Medic serves as a bridge between traditional herbal medicine and modern technology, making Ayurvedic knowledge accessible to everyone through the power of artificial intelligence. However, always consult with healthcare professionals before using any medicinal plants for treatment purposes.

🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for suggestions and improvements.

📄 License
This project is for educational and research purposes. Please ensure proper attribution when using the code or methodology.
