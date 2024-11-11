import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib

# Cargar datos
df = pd.read_csv("Vehicles.csv")  # Ajusta el nombre del archivo

# Verificar los nombres de las columnas
print("Columnas en el DataFrame:", df.columns.tolist())

# Asegurarse de que la columna 'selling_price' esté presente
if 'selling_price' not in df.columns:
    raise ValueError("La columna 'selling_price' no se encuentra en el DataFrame.")

# Identificar columnas categóricas
categorical_cols = df.select_dtypes(include=['object']).columns

# Aplicar OneHotEncoding para convertir variables categóricas en numéricas
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separar variables independientes y dependientes
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Guardar el modelo
joblib.dump(model, "modelo_precio_coche.pkl")
print("Entrenamiento completado y modelo guardado como 'modelo_precio_coche.pkl'")
