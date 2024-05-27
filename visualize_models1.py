import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

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
    layer_params = [layer.count_params() for layer in model.layers]
    layer_outputs = [layer.output_shape for layer in model.layers]

    fig, ax = plt.subplots(figsize=(15, len(layer_names) * 0.6))
    y_pos = range(len(layer_names))
    ax.barh(y_pos, layer_params, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{name} ({type})\n{output}" for name, type, output in zip(layer_names, layer_types, layer_outputs)])
    ax.invert_yaxis()  # Invertir el eje Y para que el primer layer esté en la parte superior
    ax.set_xlabel('Number of Parameters')
    ax.set_title(f'Model Architecture {index + 1}')
    
    for i in range(len(layer_params)):
        ax.text(layer_params[i], i, f'{layer_params[i]:,}', va='center', ha='left', fontsize=10)
    
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

# Función para realizar predicciones y visualizar los resultados
def predict_and_visualize(models, data, labels):
    for i, model in enumerate(models):
        predictions = model.predict(data)
        predicted_labels = np.argmax(predictions, axis=1)

        fig, ax = plt.subplots()
        ax.scatter(range(len(labels)), labels, label='True Labels', color='blue', alpha=0.6)
        ax.scatter(range(len(predicted_labels)), predicted_labels, label='Predicted Labels', color='red', alpha=0.6)
        ax.legend()
        ax.set_title(f'Predicciones del Modelo {i + 1}')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Label')
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"model_predictions_{i + 1}.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Visualización de predicciones guardada en: {plot_file}")

# Ejemplo de datos de entrada y etiquetas
# Reemplaza esto con tus datos reales
dummy_data = np.random.rand(10, 224, 224, 3)  # Ejemplo de datos de entrada (imagen 224x224x3)
dummy_labels = np.random.randint(0, 10, 10)  # Ejemplo de etiquetas

predict_and_visualize(models, dummy_data, dummy_labels)
