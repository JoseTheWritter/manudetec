import os
import cv2
import numpy as np
import json
import base64
import io
import traceback
from PIL import Image

# --- PYTORCH IMPORTS ---
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

# --- Importar rutas desde dataset_paths ---
try:
    from dataset_paths import (
        DATASET_BASE_PATH,
        DETECTOR_MODEL_PATH,
        CLASSIFIER_MODEL_PATH
    )
except ImportError:
    print("Error: No se pudo importar 'dataset_paths.py'.")
    DATASET_BASE_PATH = 'datasets'
    DETECTOR_MODEL_PATH = 'out-models/d-models'
    CLASSIFIER_MODEL_PATH = 'out-models/m-models'

"""
======================================================================
BACKEND - SERVIDOR DE PREDICCIÓN (Flask & PyTorch)
======================================================================
"""

# --- CONFIGURACIÓN DE DISPOSITIVO ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Version de PyTorch: {torch.__version__}")
print(f"Dispositivo de inferencia: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- INICIALIZACIÓN FLASK ---
app = Flask(__name__)
CORS(app)
app.config['PROPAGATE_EXCEPTIONS'] = False

# --- RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
D_MODEL_DIR_ABS = os.path.join(BASE_DIR, DETECTOR_MODEL_PATH)
M_MODEL_DIR_ABS = os.path.join(BASE_DIR, CLASSIFIER_MODEL_PATH)

# Parámetros
IMG_WIDTH = 256
IMG_HEIGHT = 256
ALLOWED_IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
DEFAULT_CATEGORY = os.environ.get("DATASET_CATEGORY")

# Cache de modelos
models_cache = {}

# --- DEFINICIÓN DE MODELOS (Deben coincidir con el entrenamiento) ---

class VGG16Autoencoder(nn.Module):
    def __init__(self):
        super(VGG16Autoencoder, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(vgg.features.children())[:24])
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)

class DefectClassifierVGG16(nn.Module):
    def __init__(self, num_classes):
        super(DefectClassifierVGG16, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)

# --- TRANSFORMACIONES ---
preprocess_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

# --- GESTIÓN DE MODELOS ---

def find_latest_model_files(directory, prefix, category=None):
    if not os.path.exists(directory): return None, None
    latest_ver = -1
    model_path, meta_path = None, None
    
    for f in os.listdir(directory):
        if f.startswith(prefix + "_") and f.endswith(".pth"):
            # Parsear versión: prefix_category_001.pth
            parts = f.replace('.pth', '').split('_')
            try:
                ver = int(parts[-1])
                cat = "_".join(parts[1:-1])
                if category and cat != category: continue
                
                if ver > latest_ver:
                    latest_ver = ver
                    model_path = os.path.join(directory, f)
                    # Buscar json asociado
                    if prefix == 'detector':
                        meta_path = os.path.join(directory, f.replace('.pth', '_threshold.json'))
                    else:
                        meta_path = os.path.join(directory, f.replace('.pth', '_labels.json'))
            except ValueError: continue
            
    return model_path, meta_path

def list_available_categories():
    cats = set()
    if not os.path.exists(D_MODEL_DIR_ABS): return []
    for f in os.listdir(D_MODEL_DIR_ABS):
        if f.startswith("detector_") and f.endswith(".pth"):
            parts = f.replace('.pth', '').split('_')
            if len(parts) >= 3:
                cats.add("_".join(parts[1:-1]))
    return sorted(list(cats))

def load_models_for_category(category):
    global models_cache
    if category in models_cache: return True

    # 1. Cargar Detector
    det_path, thresh_path = find_latest_model_files(D_MODEL_DIR_ABS, 'detector', category)
    if not det_path or not os.path.exists(thresh_path):
        print(f"Error: Faltan archivos del detector para {category}")
        return False

    # 2. Cargar Clasificador
    clf_path, lbl_path = find_latest_model_files(M_MODEL_DIR_ABS, 'classifier', category)
    if not clf_path or not os.path.exists(lbl_path):
        print(f"Error: Faltan archivos del clasificador para {category}")
        return False

    try:
        # Cargar Metadata
        with open(thresh_path, 'r') as f: threshold_data = json.load(f)
        threshold = threshold_data.get('threshold', 0.0)

        with open(lbl_path, 'r') as f: label_data = json.load(f)
        labels = {int(k): v for k,v in label_data.get('labels', {}).items()}
        num_classes = len(labels)

        # Instanciar y Cargar Detector
        detector = VGG16Autoencoder().to(DEVICE)
        detector.load_state_dict(torch.load(det_path, map_location=DEVICE))
        detector.eval()

        # Instanciar y Cargar Clasificador
        classifier = DefectClassifierVGG16(num_classes).to(DEVICE)
        classifier.load_state_dict(torch.load(clf_path, map_location=DEVICE))
        classifier.eval()

        models_cache[category] = {
            "detector": detector,
            "classifier": classifier,
            "threshold": threshold,
            "labels": labels
        }
        print(f"Modelos cargados para: {category}")
        return True
    except Exception as e:
        print(f"Error cargando modelos: {e}")
        traceback.print_exc()
        return False

def ensure_category_loaded(category):
    return load_models_for_category(category)

# --- FUNCIONES DE UTILIDAD ---

def tensor_to_numpy_image(tensor):
    # Tensor (1, C, H, W) -> Numpy (H, W, C) [0, 255] BGR para OpenCV
    img = tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0)) # C,H,W -> H,W,C
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def image_to_base64(image_cv2):
    img_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

def apply_heatmap_overlay(original, heatmap_color):
    # Asegurar tamaños iguales
    if original.shape[:2] != heatmap_color.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (original.shape[1], original.shape[0]))
    return cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

# --- RUTAS DE LA API ---

@app.route('/models/categories', methods=['GET'])
def get_categories():
    cats = list_available_categories()
    return jsonify({'categories': cats, 'default': DEFAULT_CATEGORY})

@app.route('/predict', methods=['POST'])
def predict():
    category = request.form.get('category') or DEFAULT_CATEGORY
    if not category: return jsonify({'error': 'Categoría no especificada'}), 400
    
    if not ensure_category_loaded(category):
        return jsonify({'error': f'Modelos no disponibles para {category}'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'Sin imagen'}), 400

    try:
        # 1. Cargar Imagen
        file = request.files['image']
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # 2. Preprocesar (ToTensor, Resize)
        input_tensor = preprocess_transform(pil_img).unsqueeze(0).to(DEVICE)
        
        bundle = models_cache[category]
        detector = bundle['detector']
        classifier = bundle['classifier']
        threshold = bundle['threshold']
        label_map = bundle['labels']

        # 3. Inferencia Detector (Reconstrucción)
        with torch.no_grad():
            reconstruction = detector(input_tensor)
            
            # Calcular Error MSE (igual que en entrenamiento)
            # Mean sobre (C, H, W) -> (dims 1, 2, 3)
            mse_loss = nn.MSELoss(reduction='none')(input_tensor, reconstruction)
            error_score = mse_loss.mean(dim=[1, 2, 3]).item()

        is_anomaly = error_score > threshold
        defect_type = "Sin Defecto"
        
        # 4. Generar Mapa de Calor (Visualización)
        # Diferencia absoluta promedio por canales
        diff_tensor = torch.abs(input_tensor - reconstruction).mean(dim=1, keepdim=True)
        # Normalizar a 0-255
        diff_np = diff_tensor.squeeze().cpu().detach().numpy()
        diff_norm = cv2.normalize(diff_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
        
        # Imagen original para visualización (BGR para OpenCV)
        original_cv2 = cv2.cvtColor(np.array(pil_img.resize((IMG_WIDTH, IMG_HEIGHT))), cv2.COLOR_RGB2BGR)
        
        # Cuantificación de área (Mascara binaria del heatmap)
        _, heatmap_binary = cv2.threshold(diff_norm, int(255 * 0.25), 255, cv2.THRESH_BINARY)
        area_percent = (np.count_nonzero(heatmap_binary) / diff_norm.size) * 100

        # 5. Clasificación (Solo si es anomalía)
        if is_anomaly:
            # Generar máscara para el clasificador (enfocarse en el error)
            # Usamos el heatmap binario como máscara de atención
            mask_tensor = torch.tensor(heatmap_binary, device=DEVICE).float() / 255.0
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
            
            # Aplicar máscara a la entrada (Input * Mask)
            masked_input = input_tensor * mask_tensor
            
            # Inferencia Clasificador
            with torch.no_grad():
                logits = classifier(masked_input)
                probs = torch.softmax(logits, dim=1)
                conf, class_idx = torch.max(probs, 1)
                
            class_name = label_map.get(class_idx.item(), "Desconocido")
            defect_type = f"{class_name} ({conf.item():.1%})"

        # 6. Preparar Respuesta
        # Siempre devolvemos el heatmap superpuesto
        final_viz = apply_heatmap_overlay(original_cv2, heatmap_color)
        
        return jsonify({
            'objetivo1_deteccion': bool(is_anomaly),
            'objetivo2_clasificacion': str(defect_type),
            'objetivo3_localizacion_b64': image_to_base64(final_viz),
            'objetivo4_cuantificacion_area': f"{area_percent:.2f}%",
            'debug_error': float(error_score),
            'debug_threshold': float(threshold),
            'categoria_usada': category
        })

    except Exception as e:
        print(f"Error en predicción: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# --- RUTAS DATASET (Compatibilidad con Explorador) ---
def resolve_path(category, subpath):
    clean_cat = os.path.basename(category or DEFAULT_CATEGORY or 'bottle')
    root = os.path.abspath(os.path.join(BASE_DIR, 'datasets', clean_cat, 'test'))
    target = os.path.abspath(os.path.join(root, subpath or ''))
    if not target.startswith(root): raise ValueError("Path inseguro")
    return target, root

@app.route('/dataset/list', methods=['GET'])
def list_dataset():
    cat = request.args.get('category')
    sub = request.args.get('subpath', '')
    try:
        target, root = resolve_path(cat, sub)
        if not os.path.exists(target): return jsonify({'error': 'No existe'}), 404
        
        dirs = []
        files = []
        for entry in sorted(os.listdir(target)):
            full = os.path.join(target, entry)
            rel = os.path.relpath(full, root).replace('\\', '/')
            if os.path.isdir(full):
                dirs.append({'name': entry, 'relpath': rel})
            elif entry.lower().endswith(tuple(ALLOWED_IMAGE_EXT)):
                files.append({'name': entry, 'relpath': rel})
        
        # Breadcrumbs
        crumbs = []
        if sub:
            parts = sub.replace('\\', '/').split('/')
            acc = ""
            for p in parts:
                acc = f"{acc}/{p}" if acc else p
                crumbs.append({'label': p, 'relpath': acc})

        return jsonify({
            'root': root,
            'current': sub,
            'directories': dirs,
            'files': files,
            'breadcrumbs': crumbs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/dataset/image', methods=['GET'])
def get_image():
    cat = request.args.get('category')
    path = request.args.get('path')
    try:
        target, _ = resolve_path(cat, path)
        if os.path.exists(target): return send_file(target)
        return jsonify({'error': 'No encontrado'}), 404
    except Exception:
        return jsonify({'error': 'Error'}), 400

if __name__ == '__main__':
    # Cargar categoría por defecto al inicio
    cats = list_available_categories()
    if cats:
        ensure_category_loaded(cats[0] if not DEFAULT_CATEGORY else DEFAULT_CATEGORY)
        
    app.run(debug=True, host='127.0.0.1', port=5000)