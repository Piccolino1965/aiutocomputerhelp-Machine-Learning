###############################################################
# Reti Neurali Convoluzionali - Giovanni Popolizio
# www.aiutocomputerhelp.it - 2025
###############################################################
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
# Caricamento del dataset CIFAR-10
(immagini_addestramento, etichette_addestramento), (immagini_test, etichette_test) = datasets.cifar10.load_data()
# Normalizzazione dei dati
immagini_addestramento, immagini_test = immagini_addestramento / 255.0, immagini_test / 255.0
# Classi del dataset CIFAR-10
nomi_classi = ['aereo', 'automobile', 'uccello', 'gatto', 'cervo', 'cane', 'rana', 'cavallo', 'nave', 'camion']
# Visualizzazione di alcune immagini del dataset
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(immagini_addestramento[i], cmap=plt.cm.binary)
    plt.xlabel(nomi_classi[etichette_addestramento[i][0]])
plt.show()
# Creazione del modello CNN
modello = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
# Compilazione del modello
modello.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Riepilogo del modello
modello.summary()
# Addestramento del modello
storia = modello.fit(immagini_addestramento, etichette_addestramento, epochs=10,
                    validation_data=(immagini_test, etichette_test))
# Valutazione del modello
perdita_test, accuratezza_test = modello.evaluate(immagini_test, etichette_test, verbose=2)
print(f"\nAccuratezza sul set di test: {accuratezza_test:.2f}")
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
plt.show(
