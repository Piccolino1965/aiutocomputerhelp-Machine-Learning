#############################################################################
# convoluzionali_addestramento_efficace
# giovanni popolizio 2025 www.aiutocomputerhelp.it
#############################################################################

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import matplotlib.pyplot as plt
import os
import shutil

# Pulizia della cache per forzare il download ogni volta
cache_dir = os.path.expanduser('~/.keras/datasets')
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

# Caricamento forzato del dataset CIFAR-10
(immagini_addestramento, etichette_addestramento), (immagini_test, etichette_test) = datasets.cifar10.load_data()

# Normalizzazione dei dati
immagini_addestramento, immagini_test = immagini_addestramento / 255.0, immagini_test / 255.0

# Classi del dataset CIFAR-10
nomi_classi = ['aereo', 'automobile', 'uccello', 'gatto', 'cervo', 'cane', 'rana', 'cavallo', 'nave', 'camion']

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Fit dei dati di addestramento per Data Augmentation
datagen.fit(immagini_addestramento)

# Creazione del modello CNN ottimizzato
modello = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compilazione del modello con SGD e Learning Rate Scheduler
modello.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Definizione del Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.1

lr_scheduler = LearningRateScheduler(scheduler)

# Early Stopping per evitare overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Addestramento del modello
storia = modello.fit(datagen.flow(immagini_addestramento, etichette_addestramento, batch_size=64),
                     epochs=30,
                     validation_data=(immagini_test, etichette_test),
                     callbacks=[lr_scheduler, early_stopping])

# Valutazione del modello
test_loss, test_accuracy = modello.evaluate(immagini_test, etichette_test, verbose=2)
print(f"\nAccuratezza sul set di test: {test_accuracy:.2f}")

# Visualizzazione dell'accuratezza e della perdita
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(storia.history['accuracy'], label='Accuratezza Addestramento')
plt.plot(storia.history['val_accuracy'], label='Accuratezza Validazione')
plt.title('Accuratezza durante l\'addestramento')
plt.xlabel('Epoche')
plt.ylabel('Accuratezza')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(storia.history['loss'], label='Perdita Addestramento')
plt.plot(storia.history['val_loss'], label='Perdita Validazione')
plt.title('Perdita durante l\'addestramento')
plt.xlabel('Epoche')
plt.ylabel('Perdita')
plt.legend()

plt.show()
