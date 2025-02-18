import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Configuración de directorios
melanoma_dir = 'melanoma_otsu_A'       # Reemplaza con la ruta a tu carpeta de imágenes de melanoma
no_melanoma_dir = 'no_melanoma_otsu_A' # Reemplaza con la ruta a tu carpeta de imágenes de no melanoma

# Parámetros
img_height, img_width = 224, 224
batch_size = 32
num_folds = 10
epochs = 50

# Crear el dataframe para usar con flow_from_dataframe
filepaths = []
labels = []

for filename in os.listdir(melanoma_dir):
    filepaths.append(os.path.join(melanoma_dir, filename))
    labels.append('melanoma')  # Convertir a string

for filename in os.listdir(no_melanoma_dir):
    filepaths.append(os.path.join(no_melanoma_dir, filename))
    labels.append('no_melanoma')  # Convertir a string

data = pd.DataFrame({
    'filename': filepaths,
    'class': labels
})

# Generador de imágenes
datagen = ImageDataGenerator(rescale=1./255)

# Crear el modelo InceptionV3
def create_inception_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Configurar callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Cross-validation para InceptionV3
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_no = 1
results_inception = []

for train_index, val_index in kf.split(data['filename'], data['class']):
    print(f'Fold {fold_no}')
    
    train_df = data.iloc[train_index]
    val_df = data.iloc[val_index]
    
    train_gen_split = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='class',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    val_gen_split = datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filename',
        y_col='class',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    model_inception = create_inception_model()
    
    model_checkpoint = ModelCheckpoint(f'inception_model_fold_{fold_no}.keras', save_best_only=True, monitor='val_loss')
    
    history = model_inception.fit(
        train_gen_split,
        validation_data=val_gen_split,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr, model_checkpoint]
    )
    
    results_inception.append(model_inception.evaluate(val_gen_split))
    fold_no += 1

# Resultados
print(f'Resultados de {num_folds}-Fold Cross Validation para InceptionV3:')
for i, result in enumerate(results_inception):
    print(f'Fold {i+1}: Loss = {result[0]}, Accuracy = {result[1]}')
