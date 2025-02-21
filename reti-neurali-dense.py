#########################################################
#retineuralidense.py
#www.aiutocomputerhelp.it - Giovanni Popolizio
#2025
#########################################################

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras import Input

# Generare dati sintetici
np.random.seed(72) # leggi articolo precedente sulle reti neurali
numero_campioni = 3500

# Caratteristiche: temperatura, pH, colore del terreno (categorico)
temperatura = np.random.uniform(20, 40, numero_campioni) # Temperatura tra 20 e 40 °C
ph = np.random.uniform(5, 9, numero_campioni) # pH tra 5 e 9
colori = np.random.choice(['rosso', 'verde', 'blu'], numero_campioni) # Colori categoriali

# Codifica del colore in formato one-hot
codificatore = OneHotEncoder(sparse_output=False)
colori_codificati = codificatore.fit_transform(colori.reshape(-1, 1))

# Feature complete
X = np.column_stack((temperatura, ph, colori_codificati))

# Generazione del target: tempo di replicazione (in minuti)
tempo_replicazione = 100 - (2 * temperatura) + (3 * ph) + (np.argmax(colori_codificati, axis=1) * 10) + np.random.randn(numero_campioni) * 2
tempo_replicazione = tempo_replicazione.reshape(-1, 1)

# Divisione dei dati
X_train, X_test, y_train, y_test = train_test_split(X, tempo_replicazione, test_size=0.2, random_state=42)

# Normalizzazione dei dati
scalatore = StandardScaler()
X_train = scalatore.fit_transform(X_train)
X_test = scalatore.transform(X_test)

# Creazione del modello
modello = Sequential([
Input(shape=(X_train.shape[1],)), # Definizione esplicita della forma dell'input
Dense(128, activation='relu'),
Dense(64, activation='relu'),
Dense(32, activation='relu'),
Dense(1) # Output layer per regressione
])

# Compilazione del modello
modello.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Addestramento del modello
storia = modello.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Valutazione del modello
perdita, mae = modello.evaluate(X_test, y_test)
print(f"Mean Absolute Error (MAE) sul test set: {mae}")

# Predizioni e visualizzazione
y_pred = modello.predict(X_test)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.title("Predizioni vs Valori Reali")
plt.xlabel("Valori Reali (Tempo di Replicazione)")
plt.ylabel("Predizioni")
plt.grid(True)
plt.show(block=False) # Non blocca l'esecuzione del codice

# Visualizzazione della perdita durante l'addestramento
plt.figure(figsize=(10, 6))
plt.plot(storia.history['loss'], label='Perdita sull\'addestramento')
plt.plot(storia.history['val_loss'], label='Perdita sulla validazione')
plt.title("Andamento della Perdita Durante l'Addestramento")
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show(block=False) # Non blocca l'esecuzione del codice
