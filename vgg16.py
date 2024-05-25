import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configuración de directorios
data_dir = 'path_to_dataset'  # Reemplaza con la ruta a tu dataset
melanoma_dir = os.path.join(data_dir, 'melanoma')
no_melanoma_dir = os.path.join(data_dir, 'no_melanoma')

# Parámetros
img_height, img_width = 224, 224
batch_size = 32
num_folds = 10
epochs = 50

# Generador de imágenes
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

# Cargar imágenes
train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Crear el modelo VGG16
def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Configurar callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Cross-validation
kf = KFold(n_splits=num_folds, shuffle=True)
fold_no = 1
results = []

for train_index, val_index in kf.split(train_gen.filepaths):
    print(f'Fold {fold_no}')
    
    train_files = np.array(train_gen.filepaths)[train_index]
    val_files = np.array(train_gen.filepaths)[val_index]
    
    train_gen_split = datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': train_files, 'class': train_gen.classes[train_index]}),
        directory=data_dir,
        x_col='filename',
        y_col='class',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    val_gen_split = datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': val_files, 'class': train_gen.classes[val_index]}),
        directory=data_dir,
        x_col='filename',
        y_col='class',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    model = create_model()
    
    history = model.fit(
        train_gen_split,
        validation_data=val_gen_split,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr]
    )
    
    results.append(model.evaluate(val_gen_split))
    fold_no += 1

# Resultados
print(f'Resultados de {num_folds}-Fold Cross Validation:')
for i, result in enumerate(results):
    print(f'Fold {i+1}: Loss = {result[0]}, Accuracy = {result[1]}')
