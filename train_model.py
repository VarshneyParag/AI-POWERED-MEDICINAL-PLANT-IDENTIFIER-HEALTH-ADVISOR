import cv2
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import time
from tqdm import tqdm
import warnings
from scipy.stats import skew, kurtosis
warnings.filterwarnings('ignore')

class MedicinalPlantTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
        self.scaler = StandardScaler()
        self.class_mapping = {}
        self.training_info = {}
        self.feature_cache = {}
        
    def calculate_skewness(self, data):
        """Calculate skewness manually"""
        if len(data) == 0 or np.std(data) == 0:
            return 0.0
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def calculate_kurtosis(self, data):
        """Calculate kurtosis manually"""
        if len(data) == 0 or np.std(data) == 0:
            return 0.0
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def high_accuracy_extract_features(self, image_path):
        """HIGH ACCURACY feature extraction - 200+ features for maximum accuracy"""
        try:
            # Check cache first
            if image_path in self.feature_cache:
                return self.feature_cache[image_path]
                
            # Read and resize image to HIGH resolution
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not read image: {image_path}")
                return None
                
            # HIGH RESOLUTION for maximum accuracy
            resized = cv2.resize(image, (512, 512))
            
            features = []
            
            # ===== ENHANCED COLOR FEATURES (60 features) =====
            
            # BGR color space - 10 features per channel
            for channel in range(3):
                channel_data = resized[:, :, channel].flatten()
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data),
                    np.min(channel_data),
                    np.max(channel_data),
                    np.percentile(channel_data, 10),
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75),
                    np.percentile(channel_data, 90),
                    np.var(channel_data)
                ])  # 10 × 3 = 30 features
            
            # HSV color space - 8 features per channel
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            for channel in range(3):
                channel_data = hsv[:, :, channel].flatten()
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data),
                    self.calculate_skewness(channel_data),  # Fixed: using custom function
                    self.calculate_kurtosis(channel_data),  # Fixed: using custom function
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75),
                    len(channel_data)  # Additional feature
                ])  # 8 × 3 = 24 features
            
            # LAB color space - 4 features per channel
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            for channel in range(3):
                channel_data = lab[:, :, channel].flatten()
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data),
                    np.var(channel_data)
                ])  # 4 × 3 = 12 features
            
            # Additional color moments
            for channel in range(3):
                channel_data = resized[:, :, channel].flatten()
                # Color moment 1: Mean
                # Color moment 2: Standard Deviation
                # Color moment 3: Skewness
                if len(channel_data) > 0:
                    mean = np.mean(channel_data)
                    std = np.std(channel_data)
                    skew_val = self.calculate_skewness(channel_data)  # Fixed
                    features.extend([mean, std, skew_val])
                else:
                    features.extend([0, 0, 0])
            # 3 moments × 3 channels = 9 features
            
            # Total color features: 30 + 24 + 12 + 9 = 75 features
            
            # ===== ENHANCED TEXTURE FEATURES (70 features) =====
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Multiple Sobel gradients with different kernels
            sobel_kernels = [3, 5]
            for ksize in sobel_kernels:
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
                sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
                features.extend([
                    np.mean(sobelx), np.std(sobelx), np.median(sobelx),
                    np.mean(sobely), np.std(sobely), np.median(sobely),
                    np.mean(sobel_magnitude), np.std(sobel_magnitude)
                ])  # 8 features × 2 kernels = 16 features
            
            # Multiple Laplacian filters
            laplacian_kernels = [3, 5]
            for ksize in laplacian_kernels:
                laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
                features.extend([
                    np.mean(laplacian), np.std(laplacian), np.median(laplacian),
                    np.var(laplacian)
                ])  # 4 features × 2 kernels = 8 features
            
            # Enhanced Gabor filters with multiple parameters
            gabor_params = [
                (5.0, np.pi/4, 10.0, 0.5),   # theta, lambda, sigma, gamma
                (3.0, np.pi/2, 8.0, 0.8),
                (7.0, np.pi/6, 12.0, 0.3)
            ]
            for theta, lambd, sigma, gamma in gabor_params:
                gabor_kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
                gabor_filtered = cv2.filter2D(gray, cv2.CV_64F, gabor_kernel)
                features.extend([
                    np.mean(gabor_filtered), np.std(gabor_filtered),
                    np.median(gabor_filtered), np.var(gabor_filtered)
                ])  # 4 features × 3 filters = 12 features
            
            # Canny edges with multiple thresholds
            canny_thresholds = [(50, 150), (100, 200)]
            for thresh1, thresh2 in canny_thresholds:
                edges = cv2.Canny(gray, thresh1, thresh2)
                features.extend([
                    np.mean(edges), np.std(edges), 
                    np.sum(edges > 0) / edges.size,  # Edge density
                    np.var(edges)
                ])  # 4 features × 2 thresholds = 8 features
            
            # Local Binary Pattern-like features
            lbp_features = self.compute_lbp_features(gray)
            features.extend(lbp_features)  # 8 features
            
            # Gray level co-occurrence matrix (GLCM) like features
            glcm_features = self.compute_glcm_features(gray)
            features.extend(glcm_features)  # 6 features
            
            # Additional texture statistics
            gray_flat = gray.flatten()
            features.extend([
                np.mean(gray), np.std(gray), np.median(gray), np.var(gray),
                self.calculate_skewness(gray_flat),  # Fixed
                self.calculate_kurtosis(gray_flat),  # Fixed
                np.percentile(gray, 10), np.percentile(gray, 90),
                (np.percentile(gray, 75) - np.percentile(gray, 25))  # IQR
            ])  # 9 features
            
            # Total texture features: 16 + 8 + 12 + 8 + 8 + 6 + 9 = 67 features
            
            # ===== ENHANCED SHAPE FEATURES (40 features) =====
            shape_features = []
            
            # Multiple threshold levels for robust shape detection
            thresholds = [100, 127, 150]
            for threshold in thresholds:
                _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    perimeter = cv2.arcLength(largest_contour, True)
                    hull = cv2.convexHull(largest_contour)
                    hull_area = cv2.contourArea(hull)
                    
                    if perimeter > 0:
                        shape_features.extend([
                            area, 
                            perimeter, 
                            area/perimeter,  # Compactness
                            hull_area/area if area > 0 else 0,  # Solidity
                            4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0,  # Circularity
                            len(largest_contour)  # Contour points
                        ])
                    else:
                        shape_features.extend([0, 0, 0, 0, 0, 0])
                else:
                    shape_features.extend([0, 0, 0, 0, 0, 0])
            
            # Shape features: 6 features × 3 thresholds = 18 features
            
            # Additional shape features
            shape_features.extend([
                gray.shape[1] / gray.shape[0],  # Aspect ratio
                np.sum(thresh > 0) / thresh.size if 'thresh' in locals() else 0,  # Foreground ratio
            ])  # 2 features
            
            # Total shape features: 18 + 2 = 20 features
            
            # ===== ENHANCED HISTOGRAM FEATURES (35 features) =====
            hist_features = []
            
            # Enhanced color histograms
            for channel in range(3):
                hist = cv2.calcHist([resized], [channel], None, [32], [0, 256])
                hist_features.extend([
                    np.mean(hist), 
                    np.std(hist),
                    np.median(hist),
                    np.max(hist),
                    np.argmax(hist),  # Dominant color bin
                    np.sum(hist > np.mean(hist)) / len(hist)  # Above-average ratio
                ])  # 6 features × 3 channels = 18 features
            
            # Gray histogram with enhanced features
            gray_hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist_features.extend([
                np.mean(gray_hist), 
                np.std(gray_hist),
                np.median(gray_hist),
                np.max(gray_hist),
                np.argmax(gray_hist),
                np.sum(gray_hist > 0.1 * np.max(gray_hist)) / len(gray_hist),
            ])  # 6 features
            
            # Histogram entropy
            hist_normalized = gray_hist / np.sum(gray_hist)
            hist_normalized = hist_normalized[hist_normalized > 0]
            entropy = -np.sum(hist_normalized * np.log2(hist_normalized))
            hist_features.append(entropy)  # 1 feature
            
            # Total histogram features: 18 + 6 + 1 = 25 features
            
            # ===== COMBINE ALL FEATURES =====
            all_features = np.array(features + shape_features + hist_features)
            
            # Handle NaN values
            all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Cache the result
            self.feature_cache[image_path] = all_features
            
            return all_features
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None
    
    def compute_lbp_features(self, gray):
        """Compute Local Binary Pattern features"""
        try:
            lbp_features = []
            height, width = gray.shape
            
            # Simplified LBP computation
            patterns = []
            for i in range(1, height-1, 4):  # Sample every 4th pixel for speed
                for j in range(1, width-1, 4):
                    center = gray[i,j]
                    binary_pattern = 0
                    neighbors = [
                        gray[i-1,j-1], gray[i-1,j], gray[i-1,j+1],
                        gray[i,j+1], gray[i+1,j+1], gray[i+1,j],
                        gray[i+1,j-1], gray[i,j-1]
                    ]
                    
                    for idx, neighbor in enumerate(neighbors):
                        if neighbor > center:
                            binary_pattern |= (1 << (7-idx))
                    
                    patterns.append(binary_pattern)
            
            if patterns:
                return [
                    np.mean(patterns),
                    np.std(patterns),
                    np.median(patterns),
                    np.var(patterns),
                    np.percentile(patterns, 25),
                    np.percentile(patterns, 75),
                    len(set(patterns)) / len(patterns) if patterns else 0,  # Unique patterns ratio
                    np.sum(np.array(patterns) > 127) / len(patterns) if patterns else 0,  # High value ratio
                ]
            else:
                return [0] * 8
        except:
            return [0] * 8
    
    def compute_glcm_features(self, gray):
        """Compute GLCM-like texture features"""
        try:
            # Simplified GLCM computation
            diff_x = gray[1:, :] - gray[:-1, :]  # Horizontal differences
            diff_y = gray[:, 1:] - gray[:, :-1]  # Vertical differences
            
            return [
                np.mean(np.abs(diff_x)),
                np.std(diff_x),
                np.mean(np.abs(diff_y)),
                np.std(diff_y),
                np.mean(diff_x),
                np.mean(diff_y),
            ]
        except:
            return [0] * 6
    
    def load_dataset(self):
        """Load dataset with enhanced features"""
        features = []
        labels = []
        
        print("📁 Scanning dataset directory...")
        print(f"Dataset path: {self.dataset_path}")
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset path does not exist: {self.dataset_path}")
            return [], []
        
        # Get all subdirectories (plant classes)
        plant_classes = []
        for item in os.listdir(self.dataset_path):
            item_path = os.path.join(self.dataset_path, item)
            if os.path.isdir(item_path):
                plant_classes.append(item)
        
        print(f"🌿 Found {len(plant_classes)} plant classes")
        
        # Create class mapping
        self.class_mapping = {i: class_name for i, class_name in enumerate(plant_classes)}
        
        total_images = 0
        successful_images = 0
        
        # Process each class with progress bar
        for class_idx, class_name in enumerate(plant_classes):
            class_path = os.path.join(self.dataset_path, class_name)
            
            if not os.path.exists(class_path):
                print(f"❌ Class directory not found: {class_path}")
                continue
                
            # Get all image files in the class directory
            image_files = []
            for file in os.listdir(class_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    image_files.append(file)
            
            print(f"\n📂 Processing {class_name}: {len(image_files)} images")
            
            class_images_processed = 0
            # Use tqdm for progress bar
            for image_file in tqdm(image_files, desc=f"Processing {class_name}"):
                image_path = os.path.join(class_path, image_file)
                feature_vector = self.high_accuracy_extract_features(image_path)
                
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(class_idx)
                    successful_images += 1
                    class_images_processed += 1
            
            total_images += len(image_files)
            print(f"  ✅ Successfully processed: {class_images_processed}/{len(image_files)} images")
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total images found: {total_images}")
        print(f"   Successfully processed: {successful_images}")
        if features:
            print(f"   Feature vector length: {len(features[0])}")
        else:
            print(f"   Feature vector length: 0")
        print(f"   Number of classes: {len(plant_classes)}")
        
        return features, labels
    
    def train_model(self):
        """Train the HIGH ACCURACY Random Forest model"""
        print("\n🚀 Starting HIGH ACCURACY model training for 40 plants...")
        start_time = time.time()
        
        # Load dataset
        X, y = self.load_dataset()
        
        if not X:
            print("❌ No data found for training!")
            return False
        
        if len(set(y)) < 2:
            print("❌ Need at least 2 classes for training!")
            return False
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"📦 Data shape: X={X.shape}, y={y.shape}")
        
        # Feature scaling
        print("⚙️ Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Split the data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.15, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
        
        print(f"📚 Train set: {X_train.shape[0]} samples")
        print(f"📝 Validation set: {X_val.shape[0]} samples")
        print(f"🧪 Test set: {X_test.shape[0]} samples")
        
        # HIGH ACCURACY Random Forest with optimal parameters
        print("🌳 Training HIGH ACCURACY Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=200,    # Increased for accuracy
            max_depth=25,        # Balanced depth
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,           # Use all CPU cores
            class_weight='balanced',
            oob_score=True,
            max_samples=0.9      # Use 90% of samples for each tree
        )
        
        # Train with progress indication
        print("⏳ Training in progress (this may take time for high accuracy)...")
        training_start = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - training_start
        
        # Evaluate model
        print("📈 Evaluating model...")
        
        # Training accuracy
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Validation accuracy
        y_val_pred = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        # Test accuracy
        y_test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Out-of-bag score
        oob_accuracy = self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None
        
        total_time = time.time() - start_time
        
        print(f"✅ Model trained successfully!")
        print(f"⏰ Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"⏰ Model Fitting Time: {training_time:.2f} seconds")
        print(f"🎯 Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"🎯 Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        if oob_accuracy:
            print(f"🎯 Out-of-Bag Accuracy: {oob_accuracy:.4f} ({oob_accuracy*100:.2f}%)")
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        top_features = np.sort(feature_importance)[-10:][::-1]
        print(f"🔝 Top 10 Feature Importance: {top_features}")
        
        # Save training information
        self.training_info = {
            'timestamp': datetime.now().isoformat(),
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'oob_accuracy': oob_accuracy,
            'num_classes': len(self.class_mapping),
            'classes': list(self.class_mapping.values()),
            'train_samples': X_train.shape[0],
            'val_samples': X_val.shape[0],
            'test_samples': X_test.shape[0],
            'feature_length': X.shape[1],
            'model_type': 'HIGH ACCURACY Random Forest',
            'feature_importance_top10': top_features.tolist(),
            'training_time_seconds': total_time,
            'model_fitting_time_seconds': training_time
        }
        
        # Print detailed classification report
        print("\n📊 Detailed Classification Report:")
        print(classification_report(y_test, y_test_pred, 
                                  target_names=[self.class_mapping[i] for i in range(len(self.class_mapping))]))
        
        return True
    
    def save_model(self):
        """Save the trained model and metadata"""
        if self.model is None:
            print("❌ No model to save!")
            return False
        
        try:
            # Save model
            joblib.dump(self.model, "random_forest_model.pkl")
            
            # Save scaler
            joblib.dump(self.scaler, "feature_scaler.pkl")
            
            # Save class mapping
            joblib.dump(self.class_mapping, "class_mapping.pkl")
            
            # Save training info
            joblib.dump(self.training_info, "training_info.pkl")
            
            print("💾 Model saved successfully!")
            print(f"   - Model: random_forest_model.pkl")
            print(f"   - Scaler: feature_scaler.pkl")
            print(f"   - Classes: {len(self.class_mapping)} plants")
            print(f"   - Test Accuracy: {self.training_info['test_accuracy']:.4f}")
            print(f"   - Feature Length: {self.training_info['feature_length']}")
            print(f"   - Training Time: {self.training_info['training_time_seconds']:.2f} seconds")
            print(f"   - Model Type: {self.training_info['model_type']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

def main():
    """Main training function"""
    print("🌿 HIGH ACCURACY 40 MEDICINAL PLANTS Model Trainer")
    print("=" * 60)
    print("🎯 Target: 90%+ Accuracy")
    print("🔧 Features: 180+ enhanced features")
    print("🖼️ Resolution: 512x512 pixels")
    print("🌳 Model: Enhanced Random Forest")
    print("=" * 60)
    
    # Use the provided dataset path
    dataset_path = r"C:\Users\parag\OneDrive\Desktop\AI-POWERED MEDICINAL PLANT IDENTIFIER& HEALTH ADVISOR\Medicinal plant dataset"
    
    print(f"Dataset path: {dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print("❌ Dataset path not found!")
        print("Please check the path and try again.")
        return
    
    # Initialize trainer
    trainer = MedicinalPlantTrainer(dataset_path)
    
    # Train model
    success = trainer.train_model()
    
    if success:
        # Save model
        trainer.save_model()
        
        print("\n🎉 HIGH ACCURACY training completed successfully!")
        print("🔮 You can now run 'python app.py' to start the web application")
        
        # Show model summary
        print("\n📋 Model Summary:")
        print(f"   - Classes: {len(trainer.class_mapping)}")
        print(f"   - Test Accuracy: {trainer.training_info['test_accuracy']:.2%}")
        print(f"   - Features per image: {trainer.training_info['feature_length']}")
        print(f"   - Total training samples: {trainer.training_info['train_samples']}")
        print(f"   - Total Time: {trainer.training_info['training_time_seconds']:.2f} seconds")
        print(f"   - Model Type: {trainer.training_info['model_type']}")
        
        # Accuracy assessment
        final_accuracy = trainer.training_info['test_accuracy']
        if final_accuracy >= 0.90:
            print("🎉 EXCELLENT! Achieved 90%+ accuracy! 🎉")
        elif final_accuracy >= 0.85:
            print("✅ VERY GOOD! Achieved 85%+ accuracy!")
        elif final_accuracy >= 0.80:
            print("⚠️ GOOD! Accuracy around 80%. Consider adding more training data.")
        else:
            print("❌ NEEDS IMPROVEMENT! Consider checking dataset quality.")
            
    else:
        print("\n❌ Training failed!")

if __name__ == "__main__":
    main()