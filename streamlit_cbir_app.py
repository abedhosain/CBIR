# ============================================================================
# STREAMLIT WEB INTERFACE FOR CBIR SYSTEM
# Save this as: streamlit_cbir_app.py
# Run with: streamlit run streamlit_cbir_app.py
# ============================================================================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path
import os

# Page configuration
st.set_page_config(
    page_title="Hybrid CBIR System",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #764ba2;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    FEATURES_PATH = './cbir_features'
    MODELS_PATH = './cbir_features'
    IMG_SIZE = 224
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class CNNFeatureExtractor:
    def __init__(self):
        self.device = Config.DEVICE
        self.model = models.resnet50(pretrained=False)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
    
    def extract(self, image_tensor):
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            features = self.model(image_tensor)
            features = features.view(features.size(0), -1)
            return features.cpu().numpy()[0]

class ImprovedAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, latent_dim)
        )
        
        self.latent_dim = latent_dim
    
    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc_encode(x)
        return x
    
    def forward(self, x):
        return self.encode(x)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def load_models():
    """Load CNN and Autoencoder models"""
    with st.spinner("Loading models..."):
        cnn_extractor = CNNFeatureExtractor()
        
        ae_model = ImprovedAutoencoder(latent_dim=128)
        try:
            ae_model.load_state_dict(torch.load(
                f"{Config.MODELS_PATH}/ae_128.pth",
                map_location=Config.DEVICE
            ))
            ae_model.to(Config.DEVICE)
            ae_model.eval()
        except:
            st.warning("⚠️ Autoencoder model not found. Using CNN features only.")
            ae_model = None
        
        return cnn_extractor, ae_model

@st.cache_data
def load_database_features():
    """Load pre-computed database features"""
    with st.spinner("Loading database features..."):
        try:
            with open(f"{Config.FEATURES_PATH}/train_cnn_features.pkl", 'rb') as f:
                train_cnn = pickle.load(f)
            
            with open(f"{Config.FEATURES_PATH}/train_ae_128.pkl", 'rb') as f:
                train_ae = pickle.load(f)
            
            return train_cnn, train_ae
        except FileNotFoundError:
            st.error("❌ Database features not found! Please run the training script first.")
            return None, None

def preprocess_image(image):
    """Preprocess uploaded image"""
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def compute_cosine_similarity(query_feat, db_features):
    """Compute cosine similarity"""
    query_norm = query_feat / (np.linalg.norm(query_feat) + 1e-8)
    db_norm = db_features / (np.linalg.norm(db_features, axis=1, keepdims=True) + 1e-8)
    return np.dot(db_norm, query_norm)

def compute_euclidean_distance(query_feat, db_features):
    """Compute Euclidean distance"""
    return np.linalg.norm(db_features - query_feat, axis=1)

def retrieve_images(query_feat, db_features, db_labels, db_paths, k=10, metric='cosine'):
    """Retrieve top-K similar images"""
    if metric == 'cosine':
        scores = compute_cosine_similarity(query_feat, db_features)
        top_indices = np.argsort(scores)[::-1][:k]
    else:
        scores = compute_euclidean_distance(query_feat, db_features)
        top_indices = np.argsort(scores)[:k]
    
    return {
        'indices': top_indices,
        'scores': scores[top_indices],
        'labels': db_labels[top_indices],
        'paths': [db_paths[i] for i in top_indices]
    }

def late_fusion_retrieval(query_cnn, query_ae, db_cnn, db_ae, db_labels, db_paths, 
                          k=10, w1=0.6, w2=0.4, metric='cosine'):
    """Late fusion of CNN and Autoencoder features"""
    if metric == 'cosine':
        scores_cnn = compute_cosine_similarity(query_cnn, db_cnn)
        scores_ae = compute_cosine_similarity(query_ae, db_ae)
    else:
        scores_cnn = -compute_euclidean_distance(query_cnn, db_cnn)
        scores_ae = -compute_euclidean_distance(query_ae, db_ae)
    
    scores_cnn = (scores_cnn - scores_cnn.min()) / (scores_cnn.max() - scores_cnn.min() + 1e-8)
    scores_ae = (scores_ae - scores_ae.min()) / (scores_ae.max() - scores_ae.min() + 1e-8)
    
    combined = w1 * scores_cnn + w2 * scores_ae
    top_indices = np.argsort(combined)[::-1][:k]
    
    return {
        'indices': top_indices,
        'scores': combined[top_indices],
        'labels': db_labels[top_indices],
        'paths': [db_paths[i] for i in top_indices]
    }

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🌸 Hybrid Content-Based Image Retrieval System</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Oxford Flowers Dataset - CNN + Autoencoder Features</p>', 
                unsafe_allow_html=True)
    
    # Load models and database
    cnn_extractor, ae_model = load_models()
    train_cnn, train_ae = load_database_features()
    
    if train_cnn is None or train_ae is None:
        st.stop()
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Method selection
    method = st.sidebar.selectbox(
        "Select Retrieval Method",
        ["Late Fusion (Best)", "CNN Deep Features", "Autoencoder Features"],
        help="Choose the feature extraction method"
    )
    
    # Similarity metric
    metric = st.sidebar.selectbox(
        "Similarity Metric",
        ["Cosine Similarity", "Euclidean Distance"],
        help="Choose the distance/similarity metric"
    )
    metric_type = 'cosine' if 'Cosine' in metric else 'euclidean'
    
    # Number of results
    top_k = st.sidebar.slider(
        "Number of Results (Top-K)",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        help="Number of similar images to retrieve"
    )
    
    # Fusion weights (only for Late Fusion)
    if method == "Late Fusion (Best)":
        st.sidebar.markdown("### 🔧 Fusion Weights")
        w1 = st.sidebar.slider(
            "CNN Weight (w1)",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Weight for CNN features"
        )
        w2 = 1.0 - w1
        st.sidebar.info(f"Autoencoder Weight (w2): {w2:.1f}")
    
    # Database info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Database Statistics")
    st.sidebar.metric("Total Images", len(train_cnn['labels']))
    st.sidebar.metric("Number of Classes", len(np.unique(train_cnn['labels'])))
    st.sidebar.metric("Feature Dimension (CNN)", train_cnn['deep_features'].shape[1])
    st.sidebar.metric("Feature Dimension (AE)", train_ae['features'].shape[1])
    
    # System info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💻 System Info")
    st.sidebar.info(f"Device: {Config.DEVICE}")
    st.sidebar.info(f"Models Loaded: ✅")
    
    # Main content
    st.markdown("---")
    
    # File upload
    st.markdown("## 📤 Upload Query Image")
    uploaded_file = st.file_uploader(
        "Choose a flower image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to find similar flowers in the database"
    )
    
    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🖼️ Query Image")
            query_image = Image.open(uploaded_file).convert('RGB')
            st.image(query_image, use_container_width=True, caption="Uploaded Image")
            
            # Display image info
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write(f"**Size:** {query_image.size[0]} × {query_image.size[1]}")
            st.write(f"**Format:** {query_image.format if query_image.format else 'Unknown'}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🔍 Retrieval Settings")
            
            # Show selected settings
            settings_col1, settings_col2, settings_col3 = st.columns(3)
            
            with settings_col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Method", method.split()[0])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with settings_col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Metric", metric.split()[0])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with settings_col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Top-K", top_k)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Search button
            st.markdown("")
            search_button = st.button("🔍 Search Similar Images", use_container_width=True)
        
        # Perform search
        if search_button:
            with st.spinner("🔄 Extracting features and searching..."):
                # Preprocess image
                image_tensor = preprocess_image(query_image)
                
                # Extract features based on method
                if method == "CNN Deep Features":
                    query_features = cnn_extractor.extract(image_tensor)
                    results = retrieve_images(
                        query_features,
                        train_cnn['deep_features'],
                        train_cnn['labels'],
                        train_cnn['paths'],
                        k=top_k,
                        metric=metric_type
                    )
                
                elif method == "Autoencoder Features":
                    with torch.no_grad():
                        image_tensor_batch = image_tensor.unsqueeze(0).to(Config.DEVICE)
                        query_features = ae_model(image_tensor_batch).cpu().numpy()[0]
                    
                    results = retrieve_images(
                        query_features,
                        train_ae['features'],
                        train_ae['labels'],
                        train_ae['paths'],
                        k=top_k,
                        metric=metric_type
                    )
                
                else:  # Late Fusion
                    # Extract both features
                    query_cnn_feat = cnn_extractor.extract(image_tensor)
                    
                    with torch.no_grad():
                        image_tensor_batch = image_tensor.unsqueeze(0).to(Config.DEVICE)
                        query_ae_feat = ae_model(image_tensor_batch).cpu().numpy()[0]
                    
                    results = late_fusion_retrieval(
                        query_cnn_feat,
                        query_ae_feat,
                        train_cnn['deep_features'],
                        train_ae['features'],
                        train_cnn['labels'],
                        train_cnn['paths'],
                        k=top_k,
                        w1=w1,
                        w2=w2,
                        metric=metric_type
                    )
            
            # Display results
            st.markdown("---")
            st.markdown(f"## 🎯 Top-{top_k} Similar Images")
            
            # Success message
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.write(f"✅ Found {top_k} similar images using **{method}** with **{metric}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display results in grid
            cols_per_row = 5
            num_rows = (top_k + cols_per_row - 1) // cols_per_row
            
            for row in range(num_rows):
                cols = st.columns(cols_per_row)
                
                for col_idx in range(cols_per_row):
                    img_idx = row * cols_per_row + col_idx
                    
                    if img_idx < top_k:
                        with cols[col_idx]:
                            try:
                                # Load and display image
                                result_image = Image.open(results['paths'][img_idx])
                                st.image(result_image, use_container_width=True)
                                
                                # Display info
                                score = results['scores'][img_idx]
                                label = results['labels'][img_idx]
                                rank = img_idx + 1
                                
                                # Color code based on score
                                if metric_type == 'cosine':
                                    color = "#28a745" if score > 0.7 else "#ffc107" if score > 0.5 else "#dc3545"
                                else:
                                    color = "#28a745" if score < 0.5 else "#ffc107" if score < 1.0 else "#dc3545"
                                
                                st.markdown(f"""
                                    <div style='text-align: center; padding: 0.5rem; background-color: {color}20; 
                                         border-radius: 5px; border: 2px solid {color};'>
                                        <b>Rank {rank}</b><br>
                                        Class: {label}<br>
                                        Score: {score:.4f}
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            except Exception as e:
                                st.error(f"Error loading image: {e}")
            
            # Download results option
            st.markdown("---")
            st.markdown("### 💾 Export Results")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                if st.button("📊 View Detailed Statistics"):
                    st.markdown("#### Statistics")
                    st.write(f"**Average Score:** {np.mean(results['scores']):.4f}")
                    st.write(f"**Max Score:** {np.max(results['scores']):.4f}")
                    st.write(f"**Min Score:** {np.min(results['scores']):.4f}")
                    st.write(f"**Std Dev:** {np.std(results['scores']):.4f}")
                    
                    # Class distribution
                    unique_labels, counts = np.unique(results['labels'], return_counts=True)
                    st.markdown("#### Class Distribution in Results")
                    for label, count in zip(unique_labels, counts):
                        st.write(f"Class {label}: {count} images")
            
            with export_col2:
                if st.button("📈 Show Score Distribution"):
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(range(1, top_k + 1), results['scores'], color='#667eea')
                    ax.set_xlabel('Rank', fontweight='bold')
                    ax.set_ylabel('Similarity Score', fontweight='bold')
                    ax.set_title('Similarity Scores Distribution', fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)
                    
                    st.pyplot(fig)
    
    else:
        # Instructions when no image uploaded
        st.markdown("---")
        st.info("👆 Please upload an image to start searching for similar flowers!")
        
        # Show example images from database
        st.markdown("### 📚 Sample Database Images")
        
        sample_indices = np.random.choice(len(train_cnn['labels']), min(10, len(train_cnn['labels'])), replace=False)
        
        cols = st.columns(5)
        for idx, sample_idx in enumerate(sample_indices[:5]):
            with cols[idx]:
                try:
                    img = Image.open(train_cnn['paths'][sample_idx])
                    st.image(img, use_container_width=True, caption=f"Class {train_cnn['labels'][sample_idx]}")
                except:
                    pass
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #888; padding: 2rem;'>
            <p><b>Hybrid CBIR System</b> | Oxford Flowers Dataset (20 Classes)</p>
            <p>Powered by ResNet50 + Convolutional Autoencoder | Built with Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
