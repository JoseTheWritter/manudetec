import os
import glob

"""
======================================================================
CONFIGURACION CENTRAL DE RUTAS (PATHS)
======================================================================

Instrucciones:
1.  Ajusta `DATASET_BASE_PATH` para apuntar a la carpeta raiz con tu
    dataset MVTec (o similar).
2.  Las categorias se detectan automaticamente como subcarpetas dentro
    de `DATASET_BASE_PATH`.
"""

# --- CONFIGURACION PRINCIPAL ---

# Modifica esta ruta a donde tengas tu dataset
DATASET_BASE_PATH = r'datasets'  # Ejemplo: r'C:\usuarios\tu\descargas\mvtec_ad'

# --- (NUEVO) RUTAS DE SALIDA DE MODELOS ---
MODEL_OUTPUT_BASE_PATH = r'out-models'
DETECTOR_MODEL_PATH = os.path.join(MODEL_OUTPUT_BASE_PATH, 'd-models')
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_OUTPUT_BASE_PATH, 'm-models')
# --- FIN DE NUEVO ---


# Sugerencia inicial por si no hay carpetas encontradas
DEFAULT_CATEGORY_HINT = 'bottle'


def discover_categories(base_path=None):
    """
    Lee las carpetas disponibles dentro de la ruta base del dataset.
    Cada subcarpeta se considera una categoria (ej. 'bottle', 'cable', etc).
    """
    base_path = base_path or DATASET_BASE_PATH

    if not os.path.isdir(base_path):
        print(f"ADVERTENCIA: La ruta base no existe: {base_path}")
        return []

    categories = [entry.name for entry in os.scandir(base_path) if entry.is_dir()]
    categories.sort()

    if not categories:
        print(f"ADVERTENCIA: No se encontraron categorias dentro de {base_path}")

    return categories


def pick_selected_category(available):
    """
    Elige la categoria inicial:
    1) Variable de entorno DATASET_CATEGORY.
    2) DEFAULT_CATEGORY_HINT si existe en available.
    3) La primera categoria encontrada.
    """
    env_category = os.environ.get("DATASET_CATEGORY")
    if env_category:
        return env_category

    if not available:
        return DEFAULT_CATEGORY_HINT

    if DEFAULT_CATEGORY_HINT in available:
        return DEFAULT_CATEGORY_HINT

    return available[0]


# Categorias detectadas automaticamente
AVAILABLE_CATEGORIES = discover_categories(DATASET_BASE_PATH)

# Categoria inicial sugerida (aun puede no existir en disco)
SELECTED_CATEGORY = pick_selected_category(AVAILABLE_CATEGORIES)

# --- RUTAS GENERADAS AUTOMATICAMENTE ---


class DatasetPaths:
    """
    Clase que genera y almacena todas las rutas necesarias
    basandose en la configuracion principal.
    """

    def __init__(self, base_path, category):
        if not os.path.exists(base_path):
            print(f"ADVERTENCIA: La ruta base no existe: {base_path}")
            print("Por favor, edita 'dataset_paths.py' con la ruta correcta.")

        self.base_path = base_path
        self.category = category

        # Ruta principal de la categoria seleccionada
        self.category_path = os.path.join(self.base_path, self.category)

        # 1. Ruta de Entrenamiento (Imagenes 'good')
        self.train_path = os.path.join(self.category_path, 'train', 'good')

        # 2. Ruta de Prueba (Imagenes 'good' y con defectos)
        self.test_path = os.path.join(self.category_path, 'test')

        # 3. Ruta de Mascaras (Ground Truth)
        self.ground_truth_path = os.path.join(self.category_path, 'ground_truth')

        print(f"--- Configuracion de Rutas Cargada ---")
        print(f"Categoria: {self.category}")
        print(f"Ruta de Entrenamiento: {self.train_path}")
        print(f"Ruta de Prueba: {self.test_path}")
        print(f"Ruta de Mascaras: {self.ground_truth_path}")
        print("-------------------------------------")

    def get_test_defect_folders(self):
        """
        Encuentra todas las subcarpetas dentro de 'test' que NO se llamen 'good'.
        Estas son las carpetas que contienen los defectos (ej. 'broken', 'contamination').
        """
        defect_folders = []
        if not os.path.isdir(self.test_path):
            print(f"ADVERTENCIA: La ruta de pruebas no existe: {self.test_path}")
            return defect_folders

        for folder_name in os.listdir(self.test_path):
            folder_path = os.path.join(self.test_path, folder_name)
            if os.path.isdir(folder_path) and folder_name != 'good':
                defect_folders.append(folder_name)

        if not defect_folders:
            print(f"ADVERTENCIA: No se encontraron carpetas de defectos en {self.test_path}")
            print("Asegurate de que la estructura del dataset sea correcta.")

        return defect_folders

    def get_ground_truth_mask_path(self, defect_type, img_filename):
        """
        Obtiene la ruta de la mascara correspondiente para una imagen de defecto.
        Ejemplo: .../test/broken/000.png -> .../ground_truth/broken/000_mask.png
        """
        base_filename = os.path.splitext(img_filename)[0]
        mask_filename = f"{base_filename}_mask.png"

        return os.path.join(self.ground_truth_path, defect_type, mask_filename)


def build_category_paths(categories=None, base_path=DATASET_BASE_PATH):
    """
    Crea instancias DatasetPaths para cada categoria proporcionada.
    Si no se entregan categorias, usa las detectadas automaticamente.
    """
    cats = categories or AVAILABLE_CATEGORIES
    return [DatasetPaths(base_path, cat) for cat in cats]


# --- Instancias Globales (compatibilidad con scripts existentes) ---
paths = None
try:
    # Confirmar que la categoria existe; de lo contrario, usar la primera disponible.
    chosen_category = SELECTED_CATEGORY
    if AVAILABLE_CATEGORIES and chosen_category not in AVAILABLE_CATEGORIES:
        print(f"ADVERTENCIA: La categoria '{chosen_category}' no existe en {DATASET_BASE_PATH}.")
        chosen_category = AVAILABLE_CATEGORIES[0]
        print(f"Se usara '{chosen_category}' como categoria por defecto.")

    if chosen_category:
        paths = DatasetPaths(DATASET_BASE_PATH, chosen_category)
except Exception as e:
    print(f"Error al inicializar DatasetPaths: {e}")
    print("Por favor, asegurate de que 'dataset_paths.py' este configurado correctamente.")
    paths = None


if __name__ == "__main__":
    print("\n--- Categorias Detectadas ---")
    if AVAILABLE_CATEGORIES:
        for idx, cat in enumerate(AVAILABLE_CATEGORIES, start=1):
            print(f"{idx:2d}. {cat}")
    else:
        print("No se encontraron carpetas de categorias.")
        
    print(f"\n--- Rutas de Modelos ---")
    print(f"Base de Salida: {MODEL_OUTPUT_BASE_PATH}")
    print(f"Modelos Detector: {DETECTOR_MODEL_PATH}")
    print(f"Modelos Clasificador: {CLASSIFIER_MODEL_PATH}")

    if paths:
        print("\nVerificacion de rutas:")
        print(f"Existe la carpeta de entrenamiento? {'Si' if os.path.exists(paths.train_path) else 'No'}")
        print(f"Existe la carpeta de prueba? {'Si' if os.path.exists(paths.test_path) else 'No'}")
        print(f"Existe la carpeta de mascaras? {'Si' if os.path.exists(paths.ground_truth_path) else 'No'}")

        defect_folders = paths.get_test_defect_folders()
        print(f"Carpetas de defectos encontradas: {defect_folders}")

        if defect_folders:
            test_img_path = glob.glob(os.path.join(paths.test_path, defect_folders[0], '*.png'))
            if test_img_path:
                test_img_name = os.path.basename(test_img_path[0])
                mask_path = paths.get_ground_truth_mask_path(defect_folders[0], test_img_name)
                print(f"Ejemplo de imagen: {test_img_path[0]}")
                print(f"Ejemplo de mascara: {mask_path}")
                print(f"Existe la mascara de ejemplo? {'Si' if os.path.exists(mask_path) else 'No'}")