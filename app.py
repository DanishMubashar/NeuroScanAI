import os
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import json
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import hashlib
import tempfile
import requests
from huggingface_hub import hf_hub_download, HfApi

# Import database module
from database import Database

# Page configuration
st.set_page_config(
    page_title="NeuroScanAI - Complete Brain Tumor Analysis System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== HUGGINGFACE CONFIGURATION ====================
HF_USERNAME = "DanishMubashar"  # Aapka username
MODEL_REPO_NAME = "brain-tumor-detection"  # Aapka repo name
REPO_ID = f"{HF_USERNAME}/{MODEL_REPO_NAME}"
MODEL_FILENAME = "brain_tumor_model.keras" 

# Initialize database
db = Database()

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_doctor' not in st.session_state:
    st.session_state.current_doctor = None
if 'selected_patient' not in st.session_state:
    st.session_state.selected_patient = None
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Class labels (match your training)
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (299, 299)

# ==================== HUGGINGFACE MODEL LOADER ====================

@st.cache_resource
def load_model_from_huggingface():
    """Load model directly from HuggingFace Hub"""
    
    try:
        with st.spinner("🔄 Downloading model from HuggingFace Hub..."):
            # Download model from HuggingFace
            model_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=MODEL_FILENAME,
                token=None  # Public model ke liye token nahi chahiye
            )
            
            st.success(f"✅ Model downloaded from: {REPO_ID}")
            
            # Load the model
            with st.spinner("🧠 Loading model into memory..."):
                # Try loading with different methods
                try:
                    # First try: Normal load
                    model = tf.keras.models.load_model(model_path, compile=False)
                    st.success("✅ Model loaded successfully!")
                    return model
                except Exception as e1:
                    st.warning(f"Normal loading failed, trying alternative method: {str(e1)[:100]}")
                    
                    try:
                        # Second try: With custom objects
                        custom_objects = {
                            'Functional': tf.keras.models.Model,
                            'Sequential': tf.keras.models.Sequential,
                            'BatchNormalization': tf.keras.layers.BatchNormalization
                        }
                        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                        st.success("✅ Model loaded with custom objects!")
                        return model
                    except Exception as e2:
                        st.warning(f"Custom objects failed, trying rebuild method...")
                        
                        # Third try: Rebuild model architecture
                        try:
                            # Rebuild Xception base model
                            base_model = tf.keras.applications.Xception(
                                include_top=False,
                                weights='imagenet',
                                input_shape=(299, 299, 3),
                                pooling='max'
                            )
                            
                            for layer in base_model.layers:
                                layer.trainable = False
                            
                            # Rebuild the complete model
                            rebuilt_model = tf.keras.Sequential([
                                base_model,
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dropout(0.3),
                                tf.keras.layers.Dense(128, activation='relu'),
                                tf.keras.layers.Dropout(0.25),
                                tf.keras.layers.Dense(4, activation='softmax')
                            ])
                            
                            # Try to load only the weights
                            rebuilt_model.load_weights(model_path)
                            st.success("✅ Model rebuilt and weights loaded!")
                            return rebuilt_model
                        except:
                            st.warning("Using newly built model for predictions")
                            return rebuilt_model
                            
    except Exception as e:
        st.error(f"❌ Failed to load model from HuggingFace: {str(e)}")
        st.info("""
        **Troubleshooting:**
        1. Check if the model exists at: https://huggingface.co/DanishMubashar/brain-tumor-classification
        2. Make sure the filename 'brain_tumor_model.keras' exists in the repository
        3. Try installing: pip install huggingface_hub --upgrade
        """)
        return None

@st.cache_resource
def get_model_info():
    """Get model information from HuggingFace"""
    try:
        api = HfApi()
        model_info = api.model_info(repo_id=REPO_ID)
        return {
            'name': model_info.modelId,
            'downloads': model_info.downloads,
            'likes': model_info.likes,
            'tags': model_info.tags
        }
    except:
        return None

# Load model
if not st.session_state.model_loaded:
    model = load_model_from_huggingface()
    if model is not None:
        st.session_state.model_loaded = True
        st.session_state.model = model
        st.session_state.model_info = get_model_info()
else:
    model = st.session_state.get('model', None)

# ==================== HELPER FUNCTIONS ====================

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize and normalize
        image = cv2.resize(image, IMG_SIZE)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def analyze_tumor(image, predicted_class, confidence):
    """Perform detailed tumor analysis"""
    if predicted_class == 'notumor':
        return {
            'has_tumor': False,
            'tumor_area': 0,
            'tumor_percentage': 0,
            'tumor_center': None,
            'tumor_radius': 0,
            'bounding_box': None,
            'width': 0,
            'height': 0,
            'brain_region': 'Normal',
            'hemisphere': 'Not Applicable',
            'risk_level': 'LOW'
        }
    
    try:
        # Convert image to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'has_tumor': False, 'tumor_area': 0, 'tumor_percentage': 0}
        
        # Filter contours by area
        min_area = 500
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if not valid_contours:
            return {'has_tumor': False, 'tumor_area': 0, 'tumor_percentage': 0}
        
        # Get largest contour (likely tumor)
        tumor_contour = max(valid_contours, key=cv2.contourArea)
        tumor_area = cv2.contourArea(tumor_contour)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(tumor_contour)
        
        # Calculate image area and percentage
        img_area = gray.shape[0] * gray.shape[1]
        tumor_percentage = (tumor_area / img_area) * 100
        
        # Get center
        M = cv2.moments(tumor_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = gray.shape[1] // 2, gray.shape[0] // 2
        
        # Get enclosing circle radius
        (cx, cy), radius = cv2.minEnclosingCircle(tumor_contour)
        
        # Determine brain region based on center coordinates
        img_center_x = gray.shape[1] // 2
        if center_x < img_center_x - (img_center_x * 0.2):
            hemisphere = "Right Hemisphere"
            brain_region = "Frontal/Temporal Region" if center_y < gray.shape[0] // 2 else "Parietal/Occipital Region"
        elif center_x > img_center_x + (img_center_x * 0.2):
            hemisphere = "Left Hemisphere"
            brain_region = "Frontal/Temporal Region" if center_y < gray.shape[0] // 2 else "Parietal/Occipital Region"
        else:
            hemisphere = "Midline"
            brain_region = "Central Region"
        
        # Determine risk level
        if tumor_percentage < 5:
            risk_level = "LOW"
        elif tumor_percentage < 15:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return {
            'has_tumor': True,
            'tumor_area': tumor_area,
            'tumor_percentage': tumor_percentage,
            'tumor_center': (center_x, center_y),
            'tumor_radius': int(radius),
            'bounding_box': (x, y, w, h),
            'width': w,
            'height': h,
            'brain_region': brain_region,
            'hemisphere': hemisphere,
            'risk_level': risk_level
        }
    except Exception as e:
        st.warning(f"Tumor analysis error: {e}")
        return {
            'has_tumor': predicted_class != 'notumor',
            'tumor_area': 0,
            'tumor_percentage': 0,
            'width': 0,
            'height': 0,
            'brain_region': 'Unable to determine',
            'hemisphere': 'Unable to determine',
            'risk_level': 'MEDIUM' if predicted_class != 'notumor' else 'LOW'
        }

def generate_gradcam(model, image, predicted_class_idx):
    """Generate Grad-CAM heatmap"""
    try:
        if model is None:
            return None
        
        # Try to get the base model
        if hasattr(model, 'layers') and len(model.layers) > 0:
            # Check if first layer is the base model
            if hasattr(model.layers[0], 'layers'):
                base_model = model.layers[0]
            else:
                base_model = model
        else:
            return None
        
        # Find last convolutional layer
        last_conv_layer = None
        all_layers = base_model.layers if hasattr(base_model, 'layers') else []
        
        for layer in reversed(all_layers):
            if 'conv' in layer.name and len(layer.output_shape) == 4:
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            return None
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, predicted_class_idx]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            return None
        
        # Pool gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the convolution outputs
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU and normalize
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to original image dimensions
        original_shape = (image.shape[1], image.shape[2])
        heatmap_resized = cv2.resize(heatmap, (original_shape[1], original_shape[0]))
        
        return heatmap_resized
        
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

def create_heatmap_overlay(image, heatmap):
    """Create heatmap overlay on original image"""
    if heatmap is None:
        return image
    
    try:
        # Normalize heatmap
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert image to uint8 if needed
        if image.max() <= 1.0:
            image = np.uint8(image * 255)
        
        # Ensure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))
        
        # Superimpose
        overlay = cv2.addWeighted(image, 0.6, heatmap_resized, 0.4, 0)
        
        return overlay
    except Exception as e:
        print(f"Heatmap overlay error: {e}")
        return image

# ==================== PDF REPORT GENERATION ====================

def generate_comprehensive_pdf(patient_data, visit_data, prediction_data, tumor_analysis, 
                                progression_data, original_image, gradcam_overlay, segmentation_image):
    """Generate complete PDF report with all sections"""
    
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, 
                                  textColor=colors.HexColor('#1a1a1a'), spaceAfter=12, alignment=TA_CENTER)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14,
                                    textColor=colors.HexColor('#2c3e50'), spaceAfter=10, spaceBefore=10)
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=10,
                                   textColor=colors.HexColor('#34495e'), spaceAfter=6, leading=14)
    
    # SECTION 1: Cover Page
    elements.append(Paragraph("NEUROSCAN AI", title_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("AI-Powered Brain Tumor Detection System", title_style))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                              normal_style, alignment=TA_CENTER))
    
    # Add model info
    if st.session_state.get('model_info'):
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph(f"Model: {st.session_state.model_info['name']}", 
                                  normal_style, alignment=TA_CENTER))
    
    elements.append(PageBreak())
    
    # SECTION 2: Patient Information
    elements.append(Paragraph("Patient Information", heading_style))
    patient_table_data = [
        ["Patient Name", patient_data.get('name', '')],
        ["CNIC", patient_data.get('cnic', '')],
        ["Age", f"{patient_data.get('age', '')} years"],
        ["Gender", patient_data.get('gender', '')],
        ["Contact", patient_data.get('contact', '')],
        ["Doctor", st.session_state.current_doctor.get('name', '') if st.session_state.current_doctor else 'Not Assigned'],
        ["Visit Date", visit_data.get('visit_date', datetime.now().strftime('%Y-%m-%d'))]
    ]
    patient_table = Table(patient_table_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # SECTION 3: AI Results
    elements.append(Paragraph("AI Prediction Results", heading_style))
    confidence = prediction_data.get('confidence', 0)
    tumor_type = prediction_data.get('predicted_class', 'unknown')
    
    result_table_data = [
        ["Tumor Type", tumor_type.upper()],
        ["Confidence", f"{confidence:.2f}%"],
        ["Risk Level", tumor_analysis.get('risk_level', 'N/A')],
        ["Model Used", "Xception (Transfer Learning)"]
    ]
    result_table = Table(result_table_data, colWidths=[2*inch, 4*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    elements.append(result_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # SECTION 4: Tumor Localization & Size
    if tumor_type != 'notumor':
        elements.append(Paragraph("Tumor Localization & Size Analysis", heading_style))
        
        tumor_data = [
            ["Brain Region", tumor_analysis.get('brain_region', 'N/A')],
            ["Hemisphere", tumor_analysis.get('hemisphere', 'N/A')],
            ["Tumor Area", f"{tumor_analysis.get('tumor_area', 0):.2f} pixels"],
            ["Tumor Percentage", f"{tumor_analysis.get('tumor_percentage', 0):.2f}%"],
            ["Width", f"{tumor_analysis.get('width', 0)} pixels"],
            ["Height", f"{tumor_analysis.get('height', 0)} pixels"],
            ["Tumor Center", f"{tumor_analysis.get('tumor_center', (0,0))}"],
            ["Tumor Radius", f"{tumor_analysis.get('tumor_radius', 0)} pixels"]
        ]
        tumor_table = Table(tumor_data, colWidths=[2*inch, 4*inch])
        tumor_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(tumor_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # SECTION 5: Tumor Progression (if available)
        if progression_data:
            elements.append(Paragraph("Tumor Progression Analysis", heading_style))
            
            change_color = "red" if progression_data['area_change'] > 0 else "green"
            direction_text = "increased" if progression_data['area_change'] > 0 else "decreased"
            
            progression_text = f"""
            <b>Previous Visit ({progression_data['previous_date']}):</b> {progression_data['previous_area']:.2f} pixels<br/>
            <b>Current Visit ({progression_data['current_date']}):</b> {progression_data['current_area']:.2f} pixels<br/>
            <b>Change:</b> <font color="{change_color}">{abs(progression_data['area_change']):.2f} pixels ({abs(progression_data['percentage_change']):.1f}% {direction_text})</font>
            """
            elements.append(Paragraph(progression_text, normal_style))
            elements.append(Spacer(1, 0.2*inch))
    
    # SECTION 6: Clinical Interpretation
    elements.append(Paragraph("Clinical Interpretation", heading_style))
    
    if tumor_type == 'notumor':
        interpretation = """
        The MRI scan analysis shows no evidence of brain tumor. The brain structure appears normal 
        with no abnormalities detected. Regular monitoring is recommended as per clinical protocol.
        """
    else:
        interpretation = f"""
        MRI analysis indicates a {tumor_type.upper()} tumor located in the {tumor_analysis.get('brain_region', 'brain')} 
        region ({tumor_analysis.get('hemisphere', '')}). The tumor covers approximately {tumor_analysis.get('tumor_percentage', 0):.1f}% 
        of the scanned area. Based on the size and location, this is classified as a {tumor_analysis.get('risk_level', 'MEDIUM')} 
        risk case. Immediate consultation with a neurologist is recommended for further evaluation.
        """
    
    elements.append(Paragraph(interpretation, normal_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # SECTION 7: Recommendations
    elements.append(Paragraph("Recommendations", heading_style))
    
    if tumor_type == 'notumor':
        recommendations = """
        • Continue routine health checkups
        • Follow-up MRI as clinically indicated
        • Maintain healthy lifestyle
        """
    else:
        recommendations = f"""
        • <b>Immediate:</b> Schedule consultation with a neurologist/neurosurgeon
        • <b>Diagnostic:</b> Consider additional imaging (MRI with contrast, CT scan)
        • <b>Treatment:</b> Discuss treatment options based on {tumor_type.upper()} type
        • <b>Follow-up:</b> Short-term follow-up MRI recommended within 3 months
        • <b>Biopsy:</b> Consider biopsy for definitive diagnosis if clinically indicated
        """
    
    elements.append(Paragraph(recommendations, normal_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # SECTION 8: MRI Visualizations
    elements.append(Paragraph("MRI Scan Visualizations", heading_style))
    
    if original_image is not None:
        img_buffer = BytesIO()
        original_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img = RLImage(img_buffer, width=2.5*inch, height=2.5*inch)
        
        image_elements = [img]
        
        if gradcam_overlay is not None:
            overlay_buffer = BytesIO()
            overlay_pil = Image.fromarray(cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR) if len(gradcam_overlay.shape) == 3 else gradcam_overlay)
            overlay_pil.save(overlay_buffer, format='PNG')
            overlay_buffer.seek(0)
            image_elements.append(RLImage(overlay_buffer, width=2.5*inch, height=2.5*inch))
        
        if segmentation_image is not None:
            seg_buffer = BytesIO()
            seg_pil = Image.fromarray(segmentation_image)
            seg_pil.save(seg_buffer, format='PNG')
            seg_buffer.seek(0)
            image_elements.append(RLImage(seg_buffer, width=2.5*inch, height=2.5*inch))
        
        image_table = Table([image_elements], colWidths=[2.5*inch] * len(image_elements))
        image_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
        ]))
        elements.append(image_table)
        elements.append(Spacer(1, 0.3*inch))
    
    # SECTION 9: Doctor Notes
    elements.append(Paragraph("Doctor's Notes", heading_style))
    elements.append(Paragraph("_" * 80, normal_style))
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph("_" * 80, normal_style))
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph("_" * 80, normal_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # SECTION 10: Disclaimer
    elements.append(Paragraph("Disclaimer", heading_style))
    disclaimer = """
    <i>This report is generated by an AI-powered decision support system and should not be considered as 
    a substitute for professional medical advice. The final diagnosis and treatment decisions must be made 
    by a qualified healthcare professional after comprehensive clinical evaluation.</i>
    """
    elements.append(Paragraph(disclaimer, normal_style))
    
    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer

# ==================== LOGIN UI ====================

def login_ui():
    """Doctor login interface"""
    st.markdown("""
    <style>
    .login-container {
        max-width: 450px;
        margin: 80px auto;
        padding: 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    .login-title {
        text-align: center;
        color: white;
        font-size: 32px;
        margin-bottom: 10px;
    }
    .login-subtitle {
        text-align: center;
        color: rgba(255,255,255,0.8);
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="login-title">🧠 NeuroScanAI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="login-subtitle">AI-Powered Brain Tumor Detection System</p>', unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("Login", use_container_width=True)
        
        if submitted:
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                doctor = db.authenticate_doctor(username, password)
                if doctor:
                    st.session_state.logged_in = True
                    st.session_state.current_doctor = doctor
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
    
    st.markdown('<p style="text-align:center;color:white;margin-top:20px;">Demo: dr_raj / doctor123</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== DASHBOARD PAGE ====================

def dashboard_page():
    """Main dashboard with statistics"""
    st.title("🏠 NeuroScanAI Dashboard")
    
    # Show model info if available
    if st.session_state.get('model_info'):
        with st.expander("🤖 Model Information"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", st.session_state.model_info['name'].split('/')[-1])
            with col2:
                st.metric("Downloads", st.session_state.model_info.get('downloads', 'N/A'))
            with col3:
                st.metric("Likes", st.session_state.model_info.get('likes', 'N/A'))
    
    # Get dashboard statistics
    stats = db.get_dashboard_stats()
    
    # Display stats in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", stats['total_patients'])
    with col2:
        st.metric("Total Scans", stats['total_scans'])
    with col3:
        st.metric("Tumor Cases", stats['tumor_cases'], delta="⚠️")
    with col4:
        st.metric("Today's Scans", stats['today_scans'])
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("No Tumor Cases", stats['no_tumor_cases'], delta="✅")
    with col6:
        male_count = stats['gender_stats'].get('Male', 0)
        female_count = stats['gender_stats'].get('Female', 0)
        st.metric("Male/Female", f"{male_count}/{female_count}")
    with col7:
        st.metric("Average Age", f"{stats['avg_age']} years")
    with col8:
        highest_tumor = max(stats['tumor_distribution'].items(), key=lambda x: x[1])[0] if stats['tumor_distribution'] else "None"
        st.metric("Most Common", highest_tumor.title() if highest_tumor != 'notumor' else "No Tumor")
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Tumor Type Distribution")
        if stats['tumor_distribution']:
            fig = px.pie(values=list(stats['tumor_distribution'].values()), 
                        names=list(stats['tumor_distribution'].keys()),
                        title="Distribution by Tumor Type",
                        color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tumor data available")
    
    with col2:
        st.subheader("👥 Age Group Analysis")
        if stats['age_groups']:
            age_df = pd.DataFrame(stats['age_groups'])
            fig = px.bar(age_df, x='age_group', y='count', title="Patients by Age Group",
                        color='count', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No age data available")
    
    # Charts Row 2
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📈 Monthly Scan Trends")
        if stats['monthly_trends']:
            trend_df = pd.DataFrame(stats['monthly_trends'])
            fig = px.line(trend_df, x='month', y='count', title="Scans per Month",
                         markers=True, line_shape='linear')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No scan data available")
    
    with col4:
        st.subheader("⚥ Gender Distribution")
        if stats['gender_stats']:
            fig = px.bar(x=list(stats['gender_stats'].keys()), y=list(stats['gender_stats'].values()),
                        title="Gender Distribution", color=list(stats['gender_stats'].keys()))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No gender data available")
    
    # Recent Patients
    st.markdown("---")
    st.subheader("📋 Recent Patients")
    patients = db.get_all_patients()[:10]
    if patients:
        patient_df = pd.DataFrame(patients)
        display_cols = ['name', 'cnic', 'age', 'gender', 'created_at']
        patient_df = patient_df[[c for c in display_cols if c in patient_df.columns]]
        st.dataframe(patient_df, use_container_width=True)
    else:
        st.info("No patients registered yet")

# ==================== ADD PATIENT PAGE ====================

def add_patient_page():
    """Add new patient to system"""
    st.title("👤 Add New Patient")
    
    with st.form("add_patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            cnic = st.text_input("CNIC (National ID)*", placeholder="12345-6789012-3")
            name = st.text_input("Full Name*", placeholder="Patient's full name")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
        
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            contact = st.text_input("Contact Number", placeholder="0300-1234567")
            address = st.text_area("Address", placeholder="Patient's address")
        
        submitted = st.form_submit_button("Save Patient", use_container_width=True)
        
        if submitted:
            if not cnic or not name:
                st.error("CNIC and Name are required fields!")
            else:
                # Check if patient already exists
                existing = db.get_patient_by_cnic(cnic)
                if existing:
                    st.warning(f"Patient with CNIC {cnic} already exists!")
                    st.info(f"Patient Name: {existing['name']}")
                else:
                    patient_data = {
                        'cnic': cnic,
                        'name': name,
                        'age': age,
                        'gender': gender,
                        'contact': contact,
                        'address': address
                    }
                    patient_id = db.add_patient(patient_data)
                    st.success(f"Patient {name} added successfully! (ID: {patient_id})")
                    st.session_state.selected_patient = patient_id
                    st.rerun()

# ==================== UPLOAD MRI PAGE ====================

def upload_mri_page():
    """Upload and analyze MRI scan"""
    st.title("🖼️ Upload MRI Scan for Analysis")
    
    # Select patient
    st.subheader("Select Patient")
    
    search_term = st.text_input("Search Patient (Name or CNIC)", placeholder="Type to search...")
    
    if search_term:
        patients = db.search_patients(search_term)
    else:
        patients = db.get_all_patients()[:20]
    
    if not patients:
        st.info("No patients found. Please add a patient first.")
        if st.button("➕ Add New Patient"):
            st.session_state.page = "add_patient"
            st.rerun()
        return
    
    patient_options = {f"{p['name']} (CNIC: {p['cnic']})": p['id'] for p in patients}
    selected_patient_name = st.selectbox("Choose Patient", list(patient_options.keys()))
    selected_patient_id = patient_options[selected_patient_name]
    
    patient = db.get_patient_by_id(selected_patient_id)
    
    if patient:
        st.info(f"👤 Patient: {patient['name']} | Age: {patient['age']} | Gender: {patient['gender']}")
        
        # Check previous visits
        previous_visits = db.get_patient_visits(selected_patient_id)
        if len(previous_visits) >= 1:
            st.info(f"📊 This patient has {len(previous_visits)} previous scan(s). Progression analysis will be available.")
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader("Upload MRI Scan", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True, width=300)
        
        # Doctor notes
        doctor_notes = st.text_area("Doctor's Notes (Optional)", placeholder="Add clinical notes...")
        
        # Analyze button
        if st.button("🔬 Analyze MRI Scan", type="primary", use_container_width=True):
            with st.spinner("🧠 AI is analyzing the MRI scan..."):
                # Preprocess and predict
                processed_img = preprocess_image(image)
                
                if model is None:
                    st.error("Model not loaded properly!")
                    return
                
                predictions = model.predict(processed_img)[0]
                predicted_idx = np.argmax(predictions)
                predicted_class = CLASS_LABELS[predicted_idx]
                confidence = predictions[predicted_idx] * 100
                
                # Perform tumor analysis
                img_array = np.array(image)
                tumor_analysis = analyze_tumor(img_array, predicted_class, confidence)
                
                # Generate Grad-CAM
                gradcam_heatmap = generate_gradcam(model, processed_img, predicted_idx)
                if gradcam_heatmap is not None:
                    gradcam_overlay = create_heatmap_overlay(img_array, gradcam_heatmap)
                else:
                    gradcam_overlay = None
                
                # Generate segmentation image
                if predicted_class != 'notumor':
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5,5), 0)
                    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    seg_img = img_array.copy()
                    cv2.drawContours(seg_img, contours, -1, (0,255,0), 2)
                    segmentation_image = seg_img
                else:
                    segmentation_image = img_array
                
                # Save visit to database
                visit_data = {
                    'patient_id': selected_patient_id,
                    'doctor_id': st.session_state.current_doctor['id'],
                    'mri_image_path': '',
                    'prediction_results': {'probabilities': predictions.tolist()},
                    'tumor_type': predicted_class,
                    'confidence': confidence,
                    'tumor_area': tumor_analysis.get('tumor_area', 0),
                    'tumor_width': tumor_analysis.get('width', 0),
                    'tumor_height': tumor_analysis.get('height', 0),
                    'tumor_center_x': tumor_analysis.get('tumor_center', (0,0))[0],
                    'tumor_center_y': tumor_analysis.get('tumor_center', (0,0))[1],
                    'tumor_radius': tumor_analysis.get('tumor_radius', 0),
                    'brain_region': tumor_analysis.get('brain_region', ''),
                    'hemisphere': tumor_analysis.get('hemisphere', ''),
                    'risk_level': tumor_analysis.get('risk_level', ''),
                    'doctor_notes': doctor_notes,
                    'report_pdf_path': ''
                }
                
                visit_id = db.add_visit(visit_data)
                
                # Check for progression
                progression = db.get_tumor_progression(selected_patient_id)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if predicted_class == 'notumor':
                        st.success(f"✅ Prediction: NO TUMOR DETECTED")
                    else:
                        st.error(f"⚠️ Prediction: {predicted_class.upper()} TUMOR DETECTED")
                    st.metric("Confidence Score", f"{confidence:.2f}%")
                
                with col2:
                    if predicted_class != 'notumor':
                        st.metric("Tumor Size", f"{tumor_analysis.get('tumor_percentage', 0):.1f}% of image")
                        st.metric("Risk Level", tumor_analysis.get('risk_level', 'N/A'))
                
                # Probability chart
                fig = go.Figure(data=[
                    go.Bar(x=CLASS_LABELS, y=predictions, 
                          marker_color=['red' if i == predicted_idx else 'steelblue' for i in range(len(CLASS_LABELS))])
                ])
                fig.update_layout(title="Class Probabilities", xaxis_title="Tumor Type", 
                                 yaxis_title="Probability", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tumor Analysis Details
                if predicted_class != 'notumor':
                    st.subheader("📍 Tumor Localization & Size Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Brain Region", tumor_analysis.get('brain_region', 'N/A'))
                        st.metric("Hemisphere", tumor_analysis.get('hemisphere', 'N/A'))
                    with col2:
                        st.metric("Tumor Area", f"{tumor_analysis.get('tumor_area', 0):.0f} pixels")
                        st.metric("Coverage", f"{tumor_analysis.get('tumor_percentage', 0):.1f}%")
                    with col3:
                        st.metric("Width x Height", f"{tumor_analysis.get('width', 0)} x {tumor_analysis.get('height', 0)}")
                        st.metric("Tumor Radius", f"{tumor_analysis.get('tumor_radius', 0)} pixels")
                
                # Progression Analysis
                if progression:
                    st.subheader("📈 Tumor Progression Analysis")
                    
                    change_color = "increased" if progression['area_change'] > 0 else "decreased"
                    delta_icon = "📈" if progression['area_change'] > 0 else "📉"
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Previous Visit", f"{progression['previous_area']:.0f} pixels", 
                                 help=f"Date: {progression['previous_date']}")
                    with col2:
                        st.metric("Current Visit", f"{progression['current_area']:.0f} pixels",
                                 delta=f"{delta_icon} {abs(progression['percentage_change']):.1f}% {change_color}",
                                 help=f"Date: {progression['current_date']}")
                
                # Visualizations
                st.subheader("🖼️ MRI Visualizations")
                
                viz_col1, viz_col2, viz_col3 = st.columns(3)
                
                with viz_col1:
                    st.image(image, caption="Original MRI", use_container_width=True)
                
                with viz_col2:
                    if predicted_class != 'notumor':
                        st.image(segmentation_image, caption="Tumor Segmentation", use_container_width=True)
                    else:
                        st.image(image, caption="Normal Brain", use_container_width=True)
                
                with viz_col3:
                    if gradcam_overlay is not None and predicted_class != 'notumor':
                        st.image(gradcam_overlay, caption="Grad-CAM Heatmap", use_container_width=True)
                    else:
                        st.image(image, caption="Model Focus Area", use_container_width=True)
                
                # Generate and offer PDF
                st.markdown("---")
                st.subheader("📄 Generate Report")
                
                # Prepare data for PDF
                patient_data = patient
                visit_info = {'visit_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                prediction_info = {'predicted_class': predicted_class, 'confidence': confidence}
                
                pdf_report = generate_comprehensive_pdf(
                    patient_data, visit_info, prediction_info, tumor_analysis,
                    progression, image, gradcam_overlay, segmentation_image
                )
                
                # Store current prediction for history
                st.session_state.current_prediction = {
                    'patient': patient,
                    'visit_id': visit_id,
                    'prediction': predicted_class,
                    'confidence': confidence,
                    'tumor_analysis': tumor_analysis,
                    'image': image
                }
                
                st.download_button(
                    label="📥 Download Comprehensive PDF Report",
                    data=pdf_report,
                    file_name=f"neuroscan_report_{patient['cnic']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
                st.success(f"Analysis completed! Visit ID: {visit_id}")

# ==================== PATIENT HISTORY PAGE ====================

def patient_history_page():
    """View patient history and previous scans"""
    st.title("📜 Patient History")
    
    # Search patient
    search_term = st.text_input("Search Patient (Name or CNIC)", placeholder="Type to search...")
    
    if search_term:
        patients = db.search_patients(search_term)
    else:
        patients = db.get_all_patients()[:20]
    
    if not patients:
        st.info("No patients found")
        return
    
    # Select patient
    patient_options = {f"{p['name']} (CNIC: {p['cnic']})": p for p in patients}
    selected_key = st.selectbox("Select Patient", list(patient_options.keys()))
    selected_patient = patient_options[selected_key]
    
    # Display patient info
    st.subheader(f"👤 Patient: {selected_patient['name']}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**CNIC:** {selected_patient['cnic']}")
    with col2:
        st.write(f"**Age:** {selected_patient['age']}")
    with col3:
        st.write(f"**Gender:** {selected_patient['gender']}")
    
    # Get visit history
    visits = db.get_patient_visits(selected_patient['id'])
    
    if not visits:
        st.info("No scan records found for this patient")
        return
    
    st.markdown("---")
    st.subheader(f"📊 Scan History ({len(visits)} records)")
    
    # Display visits in a table
    visit_data = []
    for visit in visits:
        visit_data.append({
            'Date': visit['visit_date'][:10] if visit['visit_date'] else 'N/A',
            'Tumor Type': visit['tumor_type'].upper() if visit['tumor_type'] else 'N/A',
            'Confidence': f"{visit['confidence']:.1f}%" if visit['confidence'] else 'N/A',
            'Risk Level': visit['risk_level'] or 'N/A',
            'Doctor': visit.get('doctor_name', 'N/A')[:20]
        })
    
    visit_df = pd.DataFrame(visit_data)
    st.dataframe(visit_df, use_container_width=True)
    
    # Progression chart for tumor patients
    tumor_visits = [v for v in visits if v['tumor_type'] != 'notumor' and v.get('tumor_area', 0) > 0]
    
    if len(tumor_visits) >= 2:
        st.subheader("📈 Tumor Size Progression Over Time")
        
        progression_data = []
        for visit in reversed(tumor_visits):
            progression_data.append({
                'Date': visit['visit_date'][:10],
                'Tumor Area (pixels)': visit['tumor_area'],
                'Tumor Type': visit['tumor_type'].upper()
            })
        
        prog_df = pd.DataFrame(progression_data)
        fig = px.line(prog_df, x='Date', y='Tumor Area (pixels)', 
                     color='Tumor Type', markers=True,
                     title="Tumor Size Progression")
        st.plotly_chart(fig, use_container_width=True)
    
    # Expandable visit details
    st.subheader("📋 Visit Details")
    for idx, visit in enumerate(visits[:5]):
        with st.expander(f"Visit {idx+1}: {visit['visit_date'][:10]} - {visit['tumor_type'].upper()}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Confidence:** {visit['confidence']:.1f}%")
                st.write(f"**Doctor:** {visit.get('doctor_name', 'N/A')}")
            with col2:
                st.write(f"**Risk Level:** {visit['risk_level'] or 'N/A'}")
                st.write(f"**Brain Region:** {visit['brain_region'] or 'N/A'}")
            
            if visit.get('doctor_notes'):
                st.write(f"**Doctor's Notes:** {visit['doctor_notes']}")

# ==================== ANALYTICS PAGE ====================

def analytics_page():
    """Advanced analytics dashboard"""
    st.title("📊 Advanced Analytics")
    
    stats = db.get_dashboard_stats()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", stats['total_patients'])
    with col2:
        st.metric("Total Scans", stats['total_scans'])
    with col3:
        tumor_rate = (stats['tumor_cases'] / stats['total_scans'] * 100) if stats['total_scans'] > 0 else 0
        st.metric("Tumor Detection Rate", f"{tumor_rate:.1f}%")
    with col4:
        st.metric("Avg Scans/Patient", f"{stats['total_scans']/stats['total_patients']:.1f}" if stats['total_patients'] > 0 else "0")
    
    st.markdown("---")
    
    # Advanced Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Tumor Detection by Type")
        if stats['tumor_distribution']:
            fig = px.pie(values=list(stats['tumor_distribution'].values()), 
                        names=list(stats['tumor_distribution'].keys()),
                        title="Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📅 Scan Distribution by Day of Week")
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT CASE CAST(strftime('%w', visit_date) AS INTEGER)
                WHEN 0 THEN 'Sunday'
                WHEN 1 THEN 'Monday'
                WHEN 2 THEN 'Tuesday'
                WHEN 3 THEN 'Wednesday'
                WHEN 4 THEN 'Thursday'
                WHEN 5 THEN 'Friday'
                WHEN 6 THEN 'Saturday'
            END as day, COUNT(*) as count
            FROM visits
            GROUP BY day
            ORDER BY MIN(CAST(strftime('%w', visit_date) AS INTEGER))
        ''')
        day_data = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        if day_data:
            day_df = pd.DataFrame(day_data)
            fig = px.bar(day_df, x='day', y='count', title="Scans per Day",
                        color='count', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    # Confidence Distribution
    st.subheader("🎯 Model Confidence Distribution")
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT confidence FROM visits WHERE confidence IS NOT NULL')
    confidences = [row['confidence'] for row in cursor.fetchall()]
    conn.close()
    
    if confidences:
        fig = px.histogram(x=confidences, nbins=20, title="Confidence Score Distribution",
                          labels={'x': 'Confidence (%)', 'y': 'Count'})
        fig.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="70% Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk Level Distribution
    st.subheader("⚠️ Risk Level Distribution")
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT risk_level, COUNT(*) as count FROM visits WHERE risk_level IS NOT NULL GROUP BY risk_level')
    risk_data = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    if risk_data:
        risk_df = pd.DataFrame(risk_data)
        colors_risk = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
        fig = px.bar(risk_df, x='risk_level', y='count', title="Risk Levels",
                    color='risk_level', color_discrete_map=colors_risk)
        st.plotly_chart(fig, use_container_width=True)

# ==================== LOGOUT ====================

def logout():
    """Logout current user"""
    st.session_state.logged_in = False
    st.session_state.current_doctor = None
    st.session_state.selected_patient = None
    st.session_state.current_prediction = None
    st.rerun()

# ==================== MAIN APP ====================

def main():
    """Main application entry point"""
    
    if not st.session_state.logged_in:
        login_ui()
        return
    
    # Show model loading status in sidebar
    st.sidebar.title("🧠 NeuroScanAI")
    st.sidebar.markdown(f"**Doctor:** {st.session_state.current_doctor['name']}")
    
    if model is None:
        st.sidebar.error("⚠️ Model not loaded!")
        if st.sidebar.button("🔄 Load Model"):
            st.cache_resource.clear()
            st.rerun()
    else:
        st.sidebar.success("✅ Model Ready")
    
    pages = {
        "Dashboard": dashboard_page,
        "Add Patient": add_patient_page,
        "Upload MRI": upload_mri_page,
        "Patient History": patient_history_page,
        "Analytics": analytics_page,
    }
    
    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Model Source:** HuggingFace")
    st.sidebar.markdown(f"`{REPO_ID}`")
    
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        logout()
    
    # Render selected page
    pages[selection]()

if __name__ == "__main__":
    main()
