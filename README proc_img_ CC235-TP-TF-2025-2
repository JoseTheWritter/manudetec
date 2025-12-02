# CC235-TP-TF-2025-2
## MANUDETECTION: Sistema Integral de Detección y Clasificación de Anomalías Industriales

## Objetivo del Trabajo

Desarrollar un sistema automatizado de inspección visual industrial que integre técnicas clásicas de procesamiento de imágenes y redes neuronales profundas para resolver cuatro objetivos críticos en control de calidad manufacturero:

1. **Detección binaria** de presencia/ausencia de defectos mediante aprendizaje de la normalidad
2. **Clasificación multiclase** del tipo específico de anomalía detectada
3. **Localización espacial** precisa de regiones defectuosas mediante mapas de calor
4. **Cuantificación de severidad** calculando el porcentaje de área comprometida

El sistema implementa una arquitectura modular de dos etapas: un autoencoder VGG16 para detección no supervisada entrenado exclusivamente con muestras normales, seguido de un clasificador VGG16 con transfer learning para discriminación de tipos de defectos. Esta aproximación permite inspección automatizada de alta velocidad (~17 imágenes/segundo en GPU) con tasas de precisión superiores al 95%, aplicable a líneas de producción industrial.

---

## Descripción del Dataset

### MVTec Anomaly Detection (MVTec AD)

El proyecto utiliza el dataset **MVTec AD**, desarrollado por MVTec Software GmbH y publicado en CVPR 2019, consolidado como el estándar de referencia para evaluación de algoritmos de detección de anomalías no supervisada en contextos industriales.

**Categorías:**

| Categoría | Tipo | Material/Textura | Defectos Típicos | Descripción | Desafíos de Detección |
|-----------|------|------------------|------------------|-------------|----------------------|
| **bottle** | Objeto | Plástico transparente/translúcido | Deformaciones, contaminación, defectos de moldeo | Botellas cilíndricas con superficies transparentes o translúcidas | Reflejos especulares complejos, transparencia variable |
| **cable** | Objeto | Plástico/caucho flexible | Dobleces, cortes, abrasión, defectos de color | Cables eléctricos con aislamiento flexible | Geometría no rígida, variabilidad de posición |
| **capsule** | Objeto | Plástico rígido farmacéutico | Grietas (crack), abolladuras (poke), rasguños (scratch) | Cápsulas farmacéuticas cilíndricas | Textura uniforme, defectos sutiles lineales |
| **hazelnut** | Objeto | Cáscara natural orgánica | Grietas, perforaciones, hendiduras | Avellanas con cáscara natural | Textura natural irregular, variabilidad orgánica |
| **metal_nut** | Objeto | Metal mecanizado | Deformaciones (bent), defectos de color, rasguños (scratch) | Tuercas metálicas hexagonales | Reflejos especulares metálicos, variación de iluminación |
| **pill** | Objeto | Fármaco comprimido | Grietas, contaminación, chips, rasguños | Píldoras farmacéuticas redondas | Forma convexa, reflejos difusos |
| **screw** | Objeto | Metal roscado | Defectos de rosca, manipulación incorrecta, rasguños | Tornillos metálicos con rosca externa | Geometría helicoidal compleja, oclusiones |
| **toothbrush** | Objeto | Plástico con cerdas | Defectos de cerdas, defectos de cabeza | Cepillos de dientes con cabezal de cerdas | Textura dual (plástico + cerdas), geometría compleja |
| **transistor** | Objeto | Componente electrónico | Defectos de soldadura, pines dañados, contaminación | Transistores electrónicos con pines metálicos | Componentes pequeños, alto detalle |
| **zipper** | Objeto | Metal/plástico articulado | Dientes rotos, defectos de tela, separación | Cremalleras con dientes metálicos/plásticos | Patrón repetitivo, articulación flexible |
| **carpet** | Textura | Tela textil | Cortes, agujeros, contaminación, hilos sueltos | Textura de alfombra con patrón regular | Patrón textural repetitivo, variabilidad de iluminación |
| **grid** | Textura | Malla metálica/plástica | Roturas de malla, defectos de glitch, dobleces | Rejilla con patrón de malla regular | Patrón geométrico estricto, detección de sutilezas |
| **leather** | Textura | Cuero natural | Pliegues, cortes, defectos de color, marcas | Superficie de cuero natural con textura irregular | Textura natural variable, defectos difusos |
| **tile** | Textura | Cerámica/azulejo | Grietas, glitch, defectos de superficie | Baldosas cerámicas con patrón regular | Superficies especulares, patrones regulares |
| **wood** | Textura | Madera natural | Grietas, agujeros, líquido, defectos de color | Textura de madera natural con vetas | Patrón natural irregular, variabilidad orgánica |

---

## Estadísticas por Categoría

| Categoría | Imágenes Train (Good) | Imágenes Test Good | Imágenes Test Defect | Total Test | Tipos de Defectos |
|-----------|----------------------|-------------------|---------------------|------------|-------------------|
| bottle | 209 | 20 | 63 | 83 | 3 tipos |
| cable | 224 | 58 | 92 | 150 | 8 tipos |
| capsule | 219 | 23 | 33 | 56 | 3 tipos |
| hazelnut | 391 | 40 | 70 | 110 | 3 tipos |
| metal_nut | 220 | 22 | 60 | 82 | 3 tipos |
| pill | 267 | 26 | 141 | 167 | 7 tipos |
| screw | 320 | 41 | 78 | 119 | 5 tipos |
| toothbrush | 60 | 12 | 30 | 42 | 1 tipo |
| transistor | 213 | 60 | 40 | 100 | 4 tipos |
| zipper | 240 | 32 | 87 | 119 | 7 tipos |
| carpet | 280 | 28 | 89 | 117 | 5 tipos |
| grid | 264 | 21 | 57 | 78 | 5 tipos |
| leather | 245 | 32 | 92 | 124 | 5 tipos |
| tile | 230 | 33 | 67 | 100 | 5 tipos |
| wood | 247 | 19 | 60 | 79 | 5 tipos |
| **TOTAL** | **3,629** | **467** | **1,059** | **1,526** | **15 categorías** |


**Características Técnicas:**
- Resolución de procesamiento: 256×256 píxeles
- Formato: PNG RGB sin compresión con pérdida
- Ground truth: Máscaras de segmentación píxel a píxel para localización precisa
- Condiciones de captura: Iluminación controlada, fondo uniforme, orientación normalizada

**Referencia:**
Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. IEEE CVPR, 9584-9592.

**Dataset disponible en:** https://www.mvtec.com/company/research/datasets/mvtec-ad

---

## Estructura del Repositorio

```
CC235-TP-TF-2025-1/
├── data/
│   └── datasets/
│       ├── capsule/
│       │   ├── train/good/          # Imágenes sin defectos (entrenamiento)
│       │   ├── test/                # Imágenes de prueba (buenas + defectos)
│       │   └── ground_truth/        # Máscaras de segmentación
│       └── metal_nut/
│           ├── train/good/
│           ├── test/
│           └── ground_truth/
│       └── [... otras 13 categorías más]

├── code/
│   ├── backend/
│   │   ├── app.py                   # Servidor Flask API
│   │   ├── dataset_paths.py         # Configuración de rutas
│   │   ├── a-train_detector.py      # Entrenamiento del detector
│   │   ├── a-train_mask.py          # Entrenamiento del clasificador
│   │   ├── requirements.txt         # Dependencias Python
│   │   └── out-models/              # Modelos entrenados (.pth)
│   └── frontend/
│       ├── index.html               # Interfaz web de inspección
│       └── style.css
└── README.md
```

---

## Tecnologías Utilizadas

### Backend (Python 3.8+)
- **PyTorch 2.x:** Framework de deep learning para entrenamiento e inferencia
- **Torchvision:** Modelos preentrenados (VGG16) y transformaciones
- **OpenCV (cv2):** Procesamiento de imágenes y generación de mapas de calor
- **Flask + Flask-CORS:** API REST para comunicación frontend-backend
- **NumPy:** Operaciones matriciales y cálculo de métricas
- **Scikit-learn:** Label encoding y división de datasets
- **Matplotlib:** Visualización de distribuciones de error

### Frontend
- **HTML5 + JavaScript (Vanilla):** Interfaz de usuario interactiva
- **Tailwind CSS:** Framework de estilos responsivos
- **Fetch API:** Comunicación asíncrona con el backend

### Hardware Utilizado
- **GPU:** NVIDIA RTX 3060 12GB (entrenamiento e inferencia)
- **CPU:** Intel i7-10700 (inferencia fallback)

---

## Metodología

### Modelo 1: Detector de Anomalías (Autoencoder VGG16)

**Técnica Clásica con Transfer Learning**

- **Arquitectura:** Encoder VGG16 congelado (pretrenado en ImageNet) + Decoder simétrico
- **Entrenamiento:** Exclusivamente con imágenes sin defectos (aprendizaje de la normalidad)
- **Detección:** Umbral estadístico sobre error MSE de reconstrucción (μ + 2σ)
- **Localización:** Mapas de calor mediante diferencia absoluta entre original y reconstrucción
- **Hiperparámetros:** 50 épocas, batch size 32, learning rate 1e-3

**Resultados:**
- Sensibilidad (Recall): 93.9% capsule, 96.7% metal_nut
- Especificidad: 95.7% capsule, 95.5% metal_nut

### Modelo 2: Clasificador de Defectos (VGG16 Fine-Tuning)

**Red Neuronal Profunda**

- **Arquitectura:** VGG16 con fine-tuning parcial (capas superiores entrenables)
- **Innovación:** Enmascaramiento selectivo de regiones defectuosas durante entrenamiento
- **Regularización:** Dropout (0.5, 0.3) + Data Augmentation agresivo
- **Hiperparámetros:** 50 épocas, batch size 32, learning rate 1e-4

**Resultados:**
- Accuracy: 95.2% capsule, 95.0% metal_nut
- Inferencia: 59ms por imagen en GPU (~17 FPS)

---

## Conclusiones

El sistema desarrollado demuestra que técnicas de deep learning preentrenado, cuidadosamente adaptadas al dominio industrial mediante estrategias como congelamiento selectivo de capas y enmascaramiento guiado, alcanzan rendimientos competitivos con el estado del arte (AUROC 0.947-0.965 vs. baseline 0.821-0.894) incluso con datasets extremadamente limitados. Las tres preguntas de investigación planteadas fueron respondidas afirmativamente:

1. **Detección no supervisada:** El autoencoder VGG16 detecta anomalías con >93% de sensibilidad entrenando solo con muestras normales, validando el enfoque de aprendizaje de la normalidad para contextos industriales con desbalance severo de clases.

2. **Clasificación de tipos:** El clasificador con fine-tuning alcanza 95% de precisión multiclase con menos de 50 muestras de entrenamiento por categoría, demostrando la efectividad del enmascaramiento selectivo para focalizar el aprendizaje en características discriminativas del defecto.

3. **Localización y cuantificación:** Los mapas de calor proporcionan localización visualmente coherente con anotaciones expertas, y la métrica de porcentaje de área afectada ofrece un indicador cuantitativo alineado con criterios industriales de aceptación.

**Limitaciones identificadas:** Falsos negativos en defectos extremadamente sutiles donde el error local se diluye al promediar sobre la imagen completa, confusión entre clases visualmente ambiguas diferenciables solo por información 3D, y dependencia de condiciones de captura controladas.

**Trabajo futuro:** Escalabilidad a las 13 categorías restantes de MVTec AD, optimización para inferencia en tiempo real (>30 FPS) en hardware embebido mediante cuantización INT8, integración de técnicas de explicabilidad (Grad-CAM, SHAP), exploración de meta-learning para adaptación rápida con <10 muestras, y segmentación semántica end-to-end mediante U-Net para localización píxel a píxel cuantificable.

El sistema representa un habilitador tecnológico para manufactura inteligente 4.0, permitiendo inspección automatizada de alta velocidad con precisión superior a inspección humana, reducción de desperdicios, y liberación de capital humano hacia actividades de mayor valor agregado como análisis de causa raíz y mejora continua de procesos.

---

## Instalación y Uso

### Requisitos Previos

- Python 3.8 o superior
- CUDA 11.x (opcional, para aceleración GPU)
- 12GB RAM mínimo (16GB recomendado)
- 2GB espacio en disco para modelos y dataset

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/usuario/CC235-TP-TF-2025-1.git
cd CC235-TP-TF-2025-1

# Instalar dependencias
cd code/backend
pip install -r requirements.txt
```

### Entrenamiento de Modelos

```bash
# 1. Entrenar detector de anomalías
python a-train_detector.py

# 2. Entrenar clasificador de defectos
python a-train_mask.py
```

Los modelos entrenados se guardarán automáticamente en `out-models/` con versionado incremental.

### Ejecución del Servidor

```bash
# Iniciar API Flask
python app.py

# El servidor estará disponible en http://127.0.0.1:5000
```

### Uso de la Interfaz Web

1. Abrir `frontend/index.html` en un navegador moderno
2. Seleccionar la categoría del modelo (capsule o metal_nut)
3. Cargar una imagen de prueba mediante drag-and-drop o explorador de archivos
4. El sistema mostrará automáticamente:
   - Estado de detección (Bueno / Defecto detectado)
   - Tipo de defecto clasificado (si aplica)
   - Mapa de calor de localización superpuesto
   - Porcentaje de área afectada

---

## Licencia

Este proyecto utiliza el dataset MVTec AD, el cual está licenciado bajo:

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**

Copyright 2019 MVTec Software GmbH

