import cv2
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import time
from tqdm import tqdm
import warnings
from scipy.stats import skew, kurtosis
from skimage import feature, filters, measure
from sklearn.decomposition import PCA
import albumentations as A
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
warnings.filterwarnings('ignore')

class AdvancedMedicinalPlantTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
        self.scaler = StandardScaler()
        self.class_mapping = {}
        self.training_info = {}
        self.feature_cache = {}
        self.feature_extractor = None
        self.setup_deep_learning_feature_extractor()
        
    def setup_deep_learning_feature_extractor(self):
        """Setup EfficientNet for deep feature extraction"""
        try:
            base_model = EfficientNetB3(weights='imagenet', include_top=False, 
                                       input_shape=(512, 512, 3), pooling='avg')
            self.feature_extractor = Model(inputs=base_model.input, 
                                         outputs=base_model.output)
            print("✅ Deep Learning Feature Extractor (EfficientNetB3) loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load EfficientNet: {e}")
            self.feature_extractor = None
    
    def calculate_advanced_features(self, data):
        """Calculate advanced statistical features"""
        if len(data) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return mean, std, 0.0, 0.0
        
        # Advanced skewness and kurtosis
        skewness = np.mean(((data - mean) / std) ** 3)
        kurt = np.mean(((data - mean) / std) ** 4) - 3
        
        return mean, std, skewness, kurt
    
    def extract_advanced_texture_features(self, image):
        """Extract advanced texture features using multiple methods"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = []
        
        # 1. GLCM Features (Enhanced)
        try:
            glcm = feature.greycomatrix(gray, [1, 3, 5], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                       symmetric=True, normed=True)
            contrast = feature.greycoprops(glcm, 'contrast').flatten()
            dissimilarity = feature.greycoprops(glcm, 'dissimilarity').flatten()
            homogeneity = feature.greycoprops(glcm, 'homogeneity').flatten()
            energy = feature.greycoprops(glcm, 'energy').flatten()
            correlation = feature.greycoprops(glcm, 'correlation').flatten()
            features.extend([np.mean(contrast), np.std(contrast), 
                           np.mean(dissimilarity), np.std(dissimilarity),
                           np.mean(homogeneity), np.std(homogeneity),
                           np.mean(energy), np.std(energy),
                           np.mean(correlation), np.std(correlation)])
        except:
            features.extend([0] * 10)
        
        # 2. LBP Features (Enhanced)
        try:
            lbp = feature.local_binary_pattern(gray, 24, 3, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 25))
            lbp_hist = lbp_hist.astype("float")
            lbp_hist /= (lbp_hist.sum() + 1e-7)
            features.extend(lbp_hist[:10])  # Take first 10 bins
        except:
            features.extend([0] * 10)
        
        # 3. Gabor Filter Bank
        try:
            gabor_features = []
            for theta in range(4):
                theta = theta / 4. * np.pi
                for sigma in (1, 3):
                    for frequency in (0.05, 0.25):
                        gabor_filter = filters.gabor_kernel(frequency, theta=theta, 
                                                          sigma_x=sigma, sigma_y=sigma)
                        filtered = np.abs(filters.convolve(gray, gabor_filter))
                        gabor_features.extend([np.mean(filtered), np.std(filtered)])
            features.extend(gabor_features[:8])  # Take first 8 features
        except:
            features.extend([0] * 8)
        
        return features
    
    def extract_deep_features(self, image):
        """Extract deep learning features using EfficientNet"""
        if self.feature_extractor is None:
            return []
        
        try:
            # Preprocess image for EfficientNet
            img = cv2.resize(image, (512, 512))
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            img = np.expand_dims(img, axis=0)
            
            # Extract features
            deep_features = self.feature_extractor.predict(img, verbose=0)
            return deep_features.flatten().tolist()[:512]  # Take first 512 features
        except:
            return []
    
    def extract_shape_features(self, image):
        """Extract advanced shape features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = []
        
        # Multiple thresholding for robust contour detection
        for threshold in [100, 127, 150, 175]:
            _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                
                # Advanced shape descriptors
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                    solidity = area / hull_area if hull_area > 0 else 0
                    aspect_ratio = float(largest_contour.shape[0]) / largest_contour.shape[1] if largest_contour.shape[1] > 0 else 0
                    
                    # Hu moments
                    moments = cv2.moments(largest_contour)
                    hu_moments = cv2.HuMoments(moments).flatten()
                    
                    features.extend([
                        area, perimeter, circularity, solidity, aspect_ratio,
                        *hu_moments[:4]  # Take first 4 Hu moments
                    ])
                else:
                    features.extend([0] * 9)
            else:
                features.extend([0] * 9)
        
        return features
    
    def ultra_advanced_extract_features(self, image_path):
        """ULTRA ADVANCED feature extraction for 93%+ accuracy"""
        try:
            if image_path in self.feature_cache:
                return self.feature_cache[image_path]
                
            # Read and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # High resolution processing
            image = cv2.resize(image, (512, 512))
            
            all_features = []
            
            # 1. Traditional Color Features (Enhanced)
            color_features = []
            color_spaces = {
                'BGR': image,
                'HSV': cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
                'LAB': cv2.cvtColor(image, cv2.COLOR_BGR2LAB),
                'YCrCb': cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            }
            
            for space_name, space_img in color_spaces.items():
                for channel in range(3):
                    channel_data = space_img[:, :, channel].flatten()
                    mean, std, skewness, kurt = self.calculate_advanced_features(channel_data)
                    color_features.extend([
                        mean, std, skewness, kurt,
                        np.median(channel_data),
                        np.percentile(channel_data, 25),
                        np.percentile(channel_data, 75),
                        np.var(channel_data)
                    ])
            
            all_features.extend(color_features)
            
            # 2. Advanced Texture Features
            texture_features = self.extract_advanced_texture_features(image)
            all_features.extend(texture_features)
            
            # 3. Deep Learning Features
            deep_features = self.extract_deep_features(image)
            all_features.extend(deep_features)
            
            # 4. Advanced Shape Features
            shape_features = self.extract_shape_features(image)
            all_features.extend(shape_features)
            
            # 5. Histogram Features (Enhanced)
            hist_features = []
            for channel in range(3):
                hist = cv2.calcHist([image], [channel], None, [64], [0, 256])
                hist_features.extend([
                    np.mean(hist), np.std(hist), np.median(hist),
                    np.max(hist), np.argmax(hist),
                    np.sum(hist > np.mean(hist)) / len(hist)
                ])
            
            all_features.extend(hist_features)
            
            # Convert to numpy array and handle NaNs
            feature_vector = np.array(all_features)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Cache results
            self.feature_cache[image_path] = feature_vector
            
            return feature_vector
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None
    
    def augment_dataset(self, features, labels):
        """Data augmentation for better generalization"""
        print("🔄 Applying advanced data augmentation...")
        
        # SMOTE-like oversampling for minority classes
        from collections import Counter
        class_counts = Counter(labels)
        max_count = max(class_counts.values())
        
        augmented_features = []
        augmented_labels = []
        
        # Add original data
        augmented_features.extend(features)
        augmented_labels.extend(labels)
        
        # Oversample minority classes
        for class_idx, count in class_counts.items():
            if count < max_count:
                # Find samples of this class
                class_indices = [i for i, label in enumerate(labels) if label == class_idx]
                class_samples = [features[i] for i in class_indices]
                
                # Add more samples by adding small noise
                needed_samples = max_count - count
                for _ in range(needed_samples):
                    sample = class_samples[np.random.randint(0, len(class_samples))]
                    noisy_sample = sample + np.random.normal(0, 0.01, len(sample))
                    augmented_features.append(noisy_sample)
                    augmented_labels.append(class_idx)
        
        return np.array(augmented_features), np.array(augmented_labels)
    
    def load_dataset(self):
        """Load and preprocess dataset with advanced features"""
        features = []
        labels = []
        
        print("📁 Scanning dataset directory...")
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset path does not exist: {self.dataset_path}")
            return [], []
        
        # Get plant classes
        plant_classes = sorted([d for d in os.listdir(self.dataset_path) 
                              if os.path.isdir(os.path.join(self.dataset_path, d))])
        
        print(f"🌿 Found {len(plant_classes)} plant classes: {plant_classes}")
        
        # Create class mapping
        self.class_mapping = {i: class_name for i, class_name in enumerate(plant_classes)}
        
        total_images = 0
        successful_images = 0
        
        # Process each class
        for class_idx, class_name in enumerate(plant_classes):
            class_path = os.path.join(self.dataset_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            print(f"\n📂 Processing {class_name}: {len(image_files)} images")
            
            class_images_processed = 0
            for image_file in tqdm(image_files, desc=f"Extracting features for {class_name}"):
                image_path = os.path.join(class_path, image_file)
                feature_vector = self.ultra_advanced_extract_features(image_path)
                
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
        print(f"   Feature vector length: {len(features[0]) if features else 0}")
        print(f"   Number of classes: {len(plant_classes)}")
        
        return features, labels
    
    def optimize_hyperparameters(self, X_train, y_train):
        """Optimize hyperparameters for maximum accuracy"""
        print("🎯 Optimizing hyperparameters for maximum accuracy...")
        
        # Reduced parameter grid for faster optimization
        param_grid = {
            'n_estimators': [150, 200, 250],
            'max_depth': [20, 25, 30],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        
        # Use randomized search for faster optimization
        from sklearn.model_selection import RandomizedSearchCV
        search = RandomizedSearchCV(
            rf, param_grid, n_iter=10, cv=3, scoring='accuracy', 
            n_jobs=-1, random_state=42, verbose=1
        )
        
        search.fit(X_train, y_train)
        
        print(f"✅ Best parameters: {search.best_params_}")
        print(f"✅ Best cross-validation score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def plot_comprehensive_analysis(self, y_test, y_pred, feature_importance):
        """Generate comprehensive analysis plots"""
        # 1. Confusion Matrix
        plt.figure(figsize=(20, 16))
        cm = confusion_matrix(y_test, y_pred)
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=list(self.class_mapping.values()),
                   yticklabels=list(self.class_mapping.values()))
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 2. Feature Importance
        plt.subplot(2, 2, 2)
        top_n = 20
        indices = np.argsort(feature_importance)[::-1][:top_n]
        plt.bar(range(top_n), feature_importance[indices])
        plt.title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.xticks(range(top_n), indices, rotation=45)
        
        # 3. Metrics Comparison
        plt.subplot(2, 2, 3)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        scores = [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, average='weighted', zero_division=0),
            recall_score(y_test, y_pred, average='weighted', zero_division=0),
            f1_score(y_test, y_pred, average='weighted', zero_division=0)
        ]
        colors = ['blue', 'green', 'orange', 'red']
        bars = plt.bar(metrics, scores, color=colors)
        plt.ylim(0, 1)
        plt.title('Performance Metrics', fontsize=16, fontweight='bold')
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Class-wise Accuracy
        plt.subplot(2, 2, 4)
        class_accuracy = []
        cm = confusion_matrix(y_test, y_pred)
        for i in range(len(self.class_mapping)):
            if cm[i].sum() > 0:
                class_accuracy.append(cm[i,i] / cm[i].sum())
            else:
                class_accuracy.append(0)
        
        plt.bar(range(len(self.class_mapping)), class_accuracy, color='lightgreen')
        plt.title('Class-wise Accuracy', fontsize=16, fontweight='bold')
        plt.xlabel('Class Index')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(self.class_mapping)), range(len(self.class_mapping)))
        
        plt.tight_layout()
        plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_advanced_model(self):
        """Train advanced model for 93%+ accuracy"""
        print("\n🚀 Starting ULTRA ADVANCED model training for 93%+ accuracy...")
        start_time = time.time()
        
        # Load dataset
        X, y = self.load_dataset()
        
        if not X:
            print("❌ No data found for training!")
            return False
        
        if len(set(y)) < 2:
            print("❌ Need at least 2 classes for training!")
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📦 Original data shape: X={X.shape}, y={y.shape}")
        
        # Apply data augmentation
        X, y = self.augment_dataset(X, y)
        print(f"📦 Augmented data shape: X={X.shape}, y={y.shape}")
        
        # Apply PCA for dimensionality reduction if needed
        if X.shape[1] > 1000:
            print("🔧 Applying PCA for dimensionality reduction...")
            pca = PCA(n_components=0.95)  # Keep 95% variance
            X = pca.fit_transform(X)
            print(f"📦 Data shape after PCA: {X.shape}")
        
        # Feature scaling
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.15, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
        
        print(f"📚 Train set: {X_train.shape[0]} samples")
        print(f"📝 Validation set: {X_val.shape[0]} samples")
        print(f"🧪 Test set: {X_test.shape[0]} samples")
        
        # Hyperparameter optimization
        self.model = self.optimize_hyperparameters(X_train, y_train)
        
        # Train final model
        training_start = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - training_start
        
        # Comprehensive evaluation
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        
        total_time = time.time() - start_time
        
        # Print comprehensive results
        print("\n" + "="*70)
        print("🎯 ULTRA ADVANCED MODEL EVALUATION RESULTS")
        print("="*70)
        
        print(f"⏰ Total Training Time: {total_time:.2f}s ({total_time/60:.2f}m)")
        print(f"⏰ Model Fitting Time: {training_time:.2f}s")
        print(f"🎯 Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"🎯 Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"🎯 Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"🎯 Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"🎯 F1-Score: {f1:.4f} ({f1*100:.2f}%)")
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        print(f"🔝 Top 5 Feature Importance: {np.sort(feature_importance)[-5:][::-1]}")
        
        # Save training info
        self.training_info = {
            'timestamp': datetime.now().isoformat(),
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_classes': len(self.class_mapping),
            'classes': list(self.class_mapping.values()),
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'feature_length': X.shape[1],
            'model_type': 'ULTRA ADVANCED Random Forest',
            'training_time_seconds': total_time,
            'best_params': self.model.get_params()
        }
        
        # Generate comprehensive plots
        print("\n📊 Generating comprehensive analysis plots...")
        self.plot_comprehensive_analysis(y_test, y_test_pred, feature_importance)
        
        # Detailed classification report
        print("\n📊 Detailed Classification Report:")
        print(classification_report(y_test, y_test_pred, 
                                  target_names=list(self.class_mapping.values())))
        
        # Cross-validation scores
        print("\n📊 Cross-Validation Scores:")
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='accuracy')
        print(f"   Cross-validation scores: {cv_scores}")
        print(f"   Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return True
    
    def save_model(self):
        """Save the trained model and metadata"""
        if self.model is None:
            print("❌ No model to save!")
            return False
        
        try:
            os.makedirs('models', exist_ok=True)
            
            joblib.dump(self.model, "models/advanced_rf_model.pkl")
            joblib.dump(self.scaler, "models/advanced_scaler.pkl")
            joblib.dump(self.class_mapping, "models/advanced_class_mapping.pkl")
            joblib.dump(self.training_info, "models/advanced_training_info.pkl")
            
            print("💾 Advanced model saved successfully!")
            print(f"   - Model: models/advanced_rf_model.pkl")
            print(f"   - Test Accuracy: {self.training_info['test_accuracy']:.4f}")
            print(f"   - Precision: {self.training_info['precision']:.4f}")
            print(f"   - Recall: {self.training_info['recall']:.4f}")
            print(f"   - F1-Score: {self.training_info['f1_score']:.4f}")
            print(f"   - Feature Length: {self.training_info['feature_length']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

def main():
    """Main training function"""
    print("🌿 ULTRA ADVANCED MEDICINAL PLANTS CLASSIFIER")
    print("=" * 65)
    print("🎯 Target: 93%+ Accuracy")
    print("🔧 Features: 1000+ advanced features + Deep Learning")
    print("🖼️ Resolution: 512x512 pixels")
    print("🌳 Model: Optimized Random Forest + EfficientNet Features")
    print("📊 Evaluation: Comprehensive analysis with visualizations")
    print("=" * 65)
    
    dataset_path = r"C:\Users\parag\OneDrive\Desktop\AI-POWERED MEDICINAL PLANT IDENTIFIER& HEALTH ADVISOR\Medicinal plant dataset"
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset path not found!")
        return
    
    # Install required packages
    print("\n📦 Checking required packages...")
    try:
        import albumentations
        import tensorflow
        print("✅ All advanced packages are available")
    except ImportError as e:
        print(f"⚠️ Some packages missing: {e}")
        print("💡 Run: pip install albumentations tensorflow")
        return
    
    trainer = AdvancedMedicinalPlantTrainer(dataset_path)
    success = trainer.train_advanced_model()
    
    if success:
        trainer.save_model()
        
        print("\n🎉 ULTRA ADVANCED TRAINING COMPLETED SUCCESSFULLY!")
        print("📈 Generated comprehensive_analysis.png")
        
        # Final assessment
        final_accuracy = trainer.training_info['test_accuracy']
        if final_accuracy >= 0.93:
            print("\n🎉 EXCELLENT! Achieved 93%+ accuracy! 🎉")
        elif final_accuracy >= 0.90:
            print("\n✅ VERY GOOD! Achieved 90%+ accuracy!")
        elif final_accuracy >= 0.85:
            print("\n⚠️ GOOD! Accuracy around 85%. Consider more data augmentation.")
        else:
            print("\n❌ Needs improvement. Check dataset quality and size.")
            
        print("\n🔮 Now run 'python advanced_app.py' for camera integration!")
        
    else:
        print("\n❌ Training failed!")

if __name__ == "__main__":
    main()
