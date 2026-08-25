# Predictor de precio de telefonos usados

Aplicacion web que predice si un telefono usado se revendera a **precio alto** o **precio bajo**,
usando un modelo de clasificacion entrenado con scikit-learn.

## Aplicacion en linea

**https://predictor-telefonos-streamlit.onrender.com/**

> Nota: el servicio esta en el plan gratuito de Render, por lo que se suspende tras unos
> minutos sin uso. La primera carga puede tardar entre 30 y 60 segundos mientras despierta.

## El problema

**Dataset:** Used Phone Price Prediction (1.000.000 de filas, 28 columnas), obtenido de Kaggle.

**Objetivo:** dadas las caracteristicas de un telefono usado (marca, RAM, almacenamiento, edad,
salud de la bateria, danios, demanda del mercado, etc.), predecir si su precio de reventa
estara por encima o por debajo de la mediana del mercado.

Es un problema de **clasificacion binaria balanceado**: al usar la mediana como punto de corte,
la mitad de los telefonos queda en cada clase.

### Nota sobre el objetivo elegido

Inicialmente se intento predecir la columna `seller_type` (Individual o Store), pero al entrenar
los modelos el ROC AUC quedo en 0.50, es decir, igual que el azar. Al analizar los datos se
comprobo que esa columna no tiene relacion con ninguna otra variable del dataset (las medias por
grupo eran practicamente identicas), por lo que fue generada de forma aleatoria y no es
predecible. Por eso se cambio el objetivo a precio alto/bajo, que si presenta senial clara.

## El modelo

Se entrenaron y compararon dos modelos, optimizando sus hiperparametros con `GridSearchCV`
(validacion cruzada de 5 pliegues, metrica ROC AUC) sobre una muestra estratificada de 50.000 filas.

| Modelo | Accuracy | ROC AUC |
|---|---|---|
| RandomForest | 0.9228 | 0.9810 |
| **AdaBoost (elegido)** | **0.9389** | **0.9889** |

**Mejores hiperparametros encontrados:**

- RandomForest: `n_estimators=280`, `criterion='gini'`, `min_samples_leaf=5`, `max_depth=12`
- AdaBoost: `n_estimators=200`, `learning_rate=1.0`

### Preparacion de los datos

- Target: `1` si el precio de reventa supera la mediana, `0` si no.
- Se excluyo `resale_price` de las variables predictoras (es el origen del target).
- Se excluyo `model` por ser redundante con `brand`.
- Las variables categoricas (`brand`, `os_type`, `condition`, `city_tier`, `seller_type`) se
  transformaron con `get_dummies(drop_first=True)`, quedando 34 columnas finales.
- Division 80/20 entre entrenamiento y validacion.

El modelo entrenado se guardo con `joblib` junto con la lista de columnas y la mediana del
precio, para poder reutilizarlo en la aplicacion web.

## Estructura del repositorio

```
.
├── app.py                    # Aplicacion web en Streamlit
├── modelo_precio.joblib      # Modelo entrenado (AdaBoost) + columnas + mediana
├── muestra_telefonos.csv     # Muestra de 20.000 filas del dataset original
├── requirements.txt          # Dependencias del proyecto
├── .python-version           # Version de Python para el despliegue
└── .gitignore
```

> El CSV original pesa 117 MB y supera el limite de 100 MB por archivo de GitHub, por lo que
> el repositorio incluye una muestra aleatoria de 20.000 filas, suficiente para los graficos
> de la aplicacion.

## Funcionalidades de la aplicacion

La app esta organizada en tres pestanias:

1. **Los datos:** metricas resumen del dataset, tabla interactiva con la cantidad de filas
   configurable y estadisticas descriptivas.
2. **Graficos:** histograma de precios con la mediana marcada, precio promedio por marca,
   precio promedio por condicion, grafico de dispersion con variable seleccionable y
   distribucion de telefonos por marca.
3. **Predecir:** formulario con todas las caracteristicas del telefono que devuelve la
   clasificacion (precio alto o bajo) junto con la probabilidad estimada.

## Como ejecutarlo localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/Ethan6966/predictor-telefonos-streamlit.git
cd predictor-telefonos-streamlit

# 2. Instalar las dependencias
pip install -r requirements.txt

# 3. Arrancar la aplicacion
streamlit run app.py
```

La aplicacion se abrira en `http://localhost:8501`.

## Despliegue en Render

El proyecto esta desplegado como Web Service en Render con la siguiente configuracion:

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- **Instance Type:** Free

## Tecnologias utilizadas

Python, pandas, scikit-learn, matplotlib, Streamlit, joblib.

## Recursos consultados

- Dataset: Used Phone Price Prediction (Kaggle)
- Documentacion de scikit-learn: `GridSearchCV`, `RandomForestClassifier`, `AdaBoostClassifier`
- Documentacion de Streamlit: widgets, layout con pestanias y columnas, y opciones del servidor
- Documentacion de Render: despliegue de servicios web en Python
