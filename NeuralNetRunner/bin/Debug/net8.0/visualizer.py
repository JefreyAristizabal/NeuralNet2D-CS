import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\PC Master\\Documents\\Another_backup\\NeuralNetProject\\NeuralNetRunner\\bin\\Debug\\net8.0\\clasificacion2d.csv")
plt.scatter(df['x'], df['y'], c=df['predicted'], cmap='coolwarm', s=10)
plt.colorbar(label='Predicción')
plt.title("Clasificación 2D por la Red Neuronal")
plt.xlabel("x")
plt.ylabel("y")
plt.show()