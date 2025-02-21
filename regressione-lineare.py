# regressionelineare_dati_generati.py
# aiutocomputerhelp.it - Giovanni Popolizio - 2025
########################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Generare il dataset
np.random.seed(42)  # Per rendere i risultati riproducibili
X = np.random.rand(100, 1) * 100  # Superficie tra 0 e 100 mq
y = 250 * X + np.random.randn(100, 1) * 1000  # Prezzo con un po' di rumore

# Step 2: Divisione dei dati
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Addestrare il modello
model = LinearRegression()
model.fit(X_train, y_train)  # Addestriamo il modello sui dati di training

# Step 4: Valutazione
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Stampiamo i risultati
coefficiente = model.coef_[0][0]
intercetta = model.intercept_[0]
print(f"Coefficiente (w): {coefficiente}")
print(f"Intercetta (b): {intercetta}")
print(f"Mean Squared Error (MSE) sul test set: {mse}")

# Step 5: Visualizzare i risultati
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Dati reali')
plt.plot(X_test, y_pred, color='red', label='Predizione')

# Aggiungere il MSE sul grafico
plt.text(5, max(y_test)[0] - 2000, f'MSE: {mse:.2f}', fontsize=12, color='green', bbox=dict(facecolor='white', alpha=0.5))

plt.title("Regressione Lineare: Superficie vs Prezzo aiutocomputerhelp.it Giovanni Popolizio")
plt.xlabel("Superficie (mq)")
plt.ylabel("Prezzo (€)")
plt.legend()
plt.grid(True)
plt.show()
