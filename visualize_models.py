import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Rutas de los modelos
model_paths = [
    "./model_fold_1.keras",
    "./model_fold_2.keras",
    "./model_fold_3.keras",
    "./model_fold_4.keras",
    "./model_fold_5.keras",
    "./model_fold_6.keras",
    "./model_fold_7.keras",
    "./model_fold_8.keras",
    "./model_fold_9.keras",
    "./model_fold_10.keras"
]

# Crear una carpeta para guardar las visualizaciones si no existe
output_dir = "model_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Verifica que los archivos existen antes de cargar los modelos
for path in model_paths:
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo {path} no existe. Verifica las rutas a los modelos.")

# Cargar los modelos con manejo de errores
models = []
for path in model_paths:
    try:
        model = tf.keras.models.load_model(path)
        models.append(model)
    except Exception as e:
        print(f"Error al cargar el modelo desde {path}: {e}")

# Función para visualizar la arquitectura del modelo
def plot_model_architecture(model, index):
    layer_names = [layer.name for layer in model.layers]
    layer_types = [layer.__class__.__name__ for layer in model.layers]

    plt.figure(figsize=(12, len(layer_names) * 0.5))
    plt.barh(layer_names, [1] * len(layer_names), tick_label=layer_types)
    plt.xlabel('Layers')
    plt.title(f'Model Architecture {index + 1}')
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"model_architecture_{index + 1}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Visualización guardada en: {plot_file}")

# Visualizar las arquitecturas
for i, model in enumerate(models):
    try:
        plot_model_architecture(model, i)
    except Exception as e:
        print(f"Error al visualizar el modelo {i+1}: {e}")

print("Visualización de modelos completada.")

