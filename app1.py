from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import cv2
import shutil
# Removed 'timm' as it's not needed for ResNet Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
# Removed 'timm' import

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ViT Model Loading (Kept for prediction, but Grad-CAM removed) ---
vit_available = False
vit_model = None
try:
    if os.path.exists("models/best_vit.pth"):
        import timm # timm is still needed for loading ViT model
        vit_model = timm.create_model('vit_base_patch16_224', pretrained=False)
        vit_model.head = torch.nn.Linear(vit_model.head.in_features, 3)
        
        state_dict = torch.load("models/best_vit.pth", map_location=device)
        vit_model.load_state_dict(state_dict)
        
        vit_model.to(device)
        vit_model.eval()
        vit_available = True
        print("✓ ViT model loaded successfully")
    else:
        print("WARNING: models/best_vit.pth not found")
        print("  ViT predictions will be unavailable")
except Exception as e:
    print(f"WARNING: Failed to load ViT model: {e}")
    print("  ViT predictions will be unavailable")
    vit_available = False
# --------------------------------------------------------------------


# Load ResNet101 model
resnet_available = False
resnet_model = None
try:
    if os.path.exists("models/best_resnet101.pth"):
        resnet_model = models.resnet101(weights=None) # Use weights=None for no pretrained weights from torchvision
        resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 3)
        
        # Try loading the state dict
        state_dict = torch.load("models/best_resnet101.pth", map_location=device)
        resnet_model.load_state_dict(state_dict)
        
        resnet_model.to(device)
        resnet_model.eval()
        resnet_available = True
        print("✓ ResNet101 model loaded successfully")
    else:
        # Changed print message to reflect the correct filename 'best_resnet101.pth'
        print("WARNING: models/best_resnet101.pth not found") 
        print("  ResNet predictions will be unavailable")
except Exception as e:
    print(f"WARNING: Failed to load ResNet model: {e}")
    print("  ResNet predictions will be unavailable")
    resnet_available = False

class_names = ["benign", "malignant", "normal"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --- Removed ViT-specific attention_rollout_visualization and vit_reshape_transform ---


def generate_gradcam_resnet(model, input_tensor, rgb_img, pred_class):
    """
    Generates Grad-CAM visualization for ResNet101.
    The target layer for ResNet-like models is typically the last convolutional block.
    For ResNet101, this is 'layer4'.
    """
    if not resnet_available:
        return None

    # Target layer for ResNet101 is the last convolutional block (layer4)
    target_layer = model.layer4[-1] 
    
    try:
        # GradCAM for ResNet does not need a reshape_transform
        cam = GradCAM(
            model=model,
            target_layers=[target_layer]
        )
        
        targets = [ClassifierOutputTarget(pred_class)]
        # Grayscale CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        
        # Simple check for uniform output
        if np.var(grayscale_cam) < 1e-5:
             print("WARNING: ResNet Grad-CAM produced uniform output.")
             return (rgb_img * 255).astype(np.uint8)

        # Post-processing the CAM
        cam_processed = grayscale_cam.copy()
        cam_processed = np.maximum(cam_processed, 0)
        
        if cam_processed.max() > 0:
            cam_processed = cam_processed / cam_processed.max()
        
        # Apply power for better visualization contrast (optional)
        cam_processed = np.power(cam_processed, 0.7) 
        
        # Overlay CAM on the original image
        cam_image = show_cam_on_image(rgb_img, cam_processed, use_rgb=True, image_weight=0.5)
        
        return cam_image
        
    except Exception as e:
        print(f"ResNet Grad-CAM failed: {e}")
        return None

def predict_with_resnet(image_tensor, rgb_img_float):
    """
    Performs prediction and generates Grad-CAM for ResNet101.
    """
    if not resnet_available:
        return None, None
    
    try:
        resnet_model.eval()
        resnet_model.zero_grad()
        
        with torch.no_grad():
            outputs = resnet_model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        pred_class = torch.argmax(probs).item()
        pred_label = class_names[pred_class]
        probs_np = probs.cpu().numpy()
        
        prediction = {
            'class': pred_label,
            'confidence': float(probs_np[pred_class]),
            'probabilities': {
                class_names[i]: float(probs_np[i]) for i in range(len(class_names))
            }
        }
        
        # Generate Grad-CAM using the new ResNet function
        cam_image = generate_gradcam_resnet(resnet_model, image_tensor, rgb_img_float, pred_class)
        
        return prediction, cam_image
    except Exception as e:
        print(f"ResNet prediction failed: {e}")
        return None, None

# Updated ViT prediction function to NOT generate Grad-CAM
def predict_with_vit(image_tensor):
    if not vit_available:
        return None
    
    try:
        vit_model.eval()
        with torch.no_grad():
            outputs = vit_model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        pred_class = torch.argmax(probs).item()
        pred_label = class_names[pred_class]
        probs_np = probs.detach().cpu().numpy()
        
        prediction = {
            'class': pred_label,
            'confidence': float(probs_np[pred_class]),
            'probabilities': {
                class_names[i]: float(probs_np[i]) for i in range(len(class_names))
            }
        }
        
        return prediction
        
    except Exception as e:
        print(f"ViT prediction failed: {e}")
        return None

def process_single_image(image_file, mask_file=None):
    try:
        filename = image_file.filename
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(image_path)
        
        img = Image.open(image_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)
        input_tensor.requires_grad = True
        
        resized_img = img.resize((224, 224))
        rgb_img_float = np.float32(resized_img) / 255.0
        
        print(f"\nProcessing: {filename}")
        
        # Get ResNet prediction and Grad-CAM
        resnet_prediction, cam_image = predict_with_resnet(input_tensor, rgb_img_float)
        if resnet_prediction:
            print(f"  ResNet: {resnet_prediction['class']} ({resnet_prediction['confidence']:.2%})")
        else:
            print(f"  ResNet: Not available")
        
        # Get ViT prediction (no Grad-CAM)
        vit_prediction = predict_with_vit(input_tensor)
        if vit_prediction:
            print(f"  ViT: {vit_prediction['class']} ({vit_prediction['confidence']:.2%})")
        else:
            print(f"  ViT: Not available")
        
        # Save Grad-CAM (from ResNet) if available
        gradcam_path = None
        if cam_image is not None:
            cam_filename = "cam_" + filename
            cam_path = os.path.join(UPLOAD_FOLDER, cam_filename)
            Image.fromarray(cam_image).save(cam_path)
            gradcam_path = f'uploads/{cam_filename}'
        
        # Save mask if provided
        mask_path = None
        if mask_file:
            mask_filename = "mask_" + mask_file.filename
            mask_path_full = os.path.join(UPLOAD_FOLDER, mask_filename)
            mask_file.save(mask_path_full)
            mask_path = f"uploads/{mask_filename}"
        
        result = {
            'original_filename': filename,
            'image_path': f'uploads/{filename}',
            'gradcam_path': gradcam_path,
            'mask_path': mask_path,
            'vit_prediction': vit_prediction,
            'resnet_prediction': resnet_prediction
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing {image_file.filename}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'original_filename': image_file.filename,
            'error': str(e)
        }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        image_files = request.files.getlist('images')
        mask_files = request.files.getlist('masks')
        
        if not image_files or image_files[0].filename == '':
            return jsonify({'success': False, 'error': 'No images provided'})
        
        print(f"\n{'='*60}")
        print(f"Processing {len(image_files)} image(s)")
        print(f"{'='*60}")
        
        results = []
        
        for i, image_file in enumerate(image_files):
            mask_file = mask_files[i] if i < len(mask_files) else None
            result = process_single_image(image_file, mask_file)
            results.append(result)
        
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route("/clear", methods=["POST"])
def clear():
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        return jsonify({'success': True, 'message': 'All files cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Breast Cancer Classification System")
    print(f"Device: {device}")
    print(f"ResNet101: {'Available' if resnet_available else 'Not Available'}")
    print(f"ViT: {'Available' if vit_available else 'Not Available'}")
    print("="*60 + "\n")
    
    if not resnet_available and not vit_available:
        print("ERROR: No models available! Please ensure model files are in 'models/' directory")
        print("  - models/best_resnet101.pth")
        print("  - models/best_vit.pth")
    
    app.run(debug=True)