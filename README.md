# Titanic: Machine Learning for Survival Prediction 🚢

Este es mi primer proyecto de Ciencia de Datos de principio a fin, basado en la clásica competición de [Kaggle "Titanic - Machine Learning from Disaster"](https://www.kaggle.com/c/titanic).

El objetivo es construir un modelo de Machine Learning que prediga qué pasajeros sobrevivieron al desastre del Titanic basándose en un conjunto de características.

---

##  workflow del Proyecto

Mi proceso se dividió en 6 fases clave, documentadas en el notebook `titanic-analysis.ipynb`:

### 1. Carga e Inspección
* Carga de los `train.csv` y `test.csv`.
* Inspección inicial (`.info()`, `.describe()`) para identificar tipos de datos, valores nulos (en `Age`, `Cabin`, `Embarked`) y estadísticas básicas.

### 2. Análisis Exploratorio de Datos (EDA)
* Análisis de correlación inicial usando `groupby()` para ver la tasa de supervivencia por `Pclass`, `Sex`, `SibSp` y `Parch`.
* Visualizaciones con `seaborn` para entender las distribuciones (ej. `Age vs. Survival`).

### 3. Ingeniería de Características (El Trabajo de "Detective")
Esta fue la fase más crítica. Creé varias características nuevas para mejorar la precisión del modelo:
* `**Family_Size`**: Combinando `SibSp` y `Parch`.
* `**Family_Size_Grouped`**: Agrupando `Family_Size` en categorías útiles ('Alone', 'Small', 'Medium', 'Large').
* `**Age_Cut`**: Binarizando la columna `Age` en 8 grupos basados en `pd.qcut()`.
* `**Fare_Cut`**: Binarizando la columna `Fare` (muy sesgada) en 6 grupos.
* `**Title`**: Extraído de la columna `Name` (ej. 'Mr', 'Mrs', 'Master') y agrupando títulos raros ('Military', 'Noble').
* `**TicketNumberCounts`**: Calculando cuántos pasajeros compartían el mismo número de ticket.
* `**Cabin_Assigned`**: Una característica binaria (1 o 0) que indica si un pasajero tenía una `Cabin` asignada o era 'U' (Desconocido).

### 4. Preprocesamiento (La Pipeline)
* **Limpieza Final:** Rellené los últimos valores nulos (ej. `Embarked` con la moda 'S', `Fare` con la mediana).
* **Eliminación de Columnas:** Eliminé las columnas "crudas" que ya no eran necesarias (ej. `Name`, `Ticket`, `SibSp`).
* **Pipeline:** Construí un `ColumnTransformer` para automatizar todo el preprocesamiento, incluyendo:
    * `SimpleImputer()` para rellenar cualquier nulo restante.
    * `OneHotEncoder()` para convertir todas las características categóricas (ej. `Sex`, `Title`) en números.

### 5. Entrenamiento y Optimización de Modelos
* Dividí los datos en `X_train`, `y_train` y `X_valid` para la validación.
* **Comparé 6 modelos de clasificación diferentes**:
    1.  Random Forest
    2.  Decision Tree
    3.  K-Neighbors (KNN)
    4.  Support Vector (SVC)
    5.  Logistic Regression
    6.  Gaussian Naive Bayes
* Usé `GridSearchCV` con `StratifiedKFold(n_splits=5)` en cada modelo para encontrar los mejores hiperparámetros y prevenir el sobreajuste.

### 6. Resultados y Envío
* Comparé las puntuaciones de *accuracy* de los 6 modelos optimizados.
* El modelo com mayor *accuracy* es *Random Forest*.
* Generé los archivos `submission[i].csv`.

---

## Resultados
La fase de *Feature Engineering* fue clave. Características como `Title` y `Cabin_Assigned` demostraron ser predictores muy fuertes.

| Modelo | Mejor Accuracy (Cross-Validation) |
| :--- | :--- |
| **Random Forest** | **[0.83]** |
| Decision Tree | [0.8159] |
| K-Neighbors (KNN) | [0.8076] |
| Support Vector (SVC)| [0.7991] |
| Logistic Regression | [0.8048] |
| Gaussian Naive Bayes| [0.7795] |

---

## Cómo Ejecutar
1.  Clona o descarga este repositorio.
2.  Asegúrate de tener las librerías necesarias (`pandas`, `numpy`, `sklearn`, `seaborn`, `matplotlib`).
3.  Abre `titanic-analysis.ipynb` en Jupyter Notebook y haz clic en "Kernel" -> "Restart & Run All".
