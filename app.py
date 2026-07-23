# -*- coding: utf-8 -*-
"""
App en Streamlit para explorar el dataset de telefonos usados y predecir
si un telefono se revendera a precio ALTO o BAJO.
Usa el modelo entrenado en el paso 2 (modelo_precio.joblib).
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# --- Configuracion de la pagina (siempre va primero) ---
st.set_page_config(page_title="Telefonos usados", page_icon="📱", layout="wide")


# --- Cargar datos y modelo ---
# El decorador @st.cache_data hace que el CSV se lea una sola vez y no en cada clic.
@st.cache_data
def cargar_datos():
    return pd.read_csv("muestra_telefonos.csv")


@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_precio.joblib")


df = cargar_datos()
artefacto = cargar_modelo()

modelo = artefacto["modelo"]
COLUMNAS = artefacto["columnas"]        # las 34 columnas que espera el modelo
CLASES = artefacto["clases"]            # {0: 'Precio bajo', 1: 'Precio alto'}
MEDIANA = artefacto["mediana_precio"]   # el corte entre alto y bajo
NOMBRE_MODELO = artefacto["nombre"]     # AdaBoost o RandomForest

CATEGORICAS = ["brand", "os_type", "condition", "city_tier", "seller_type"]


# --- Titulo ---
st.title("📱 Telefonos usados: explorar y predecir")
st.write(
    f"Modelo entrenado: **{NOMBRE_MODELO}**. "
    f"Un telefono es de *precio alto* si se revende por encima de la mediana "
    f"(**{MEDIANA:,.2f}**)."
)

# --- Tres pestanias ---
tab_datos, tab_graficos, tab_prediccion = st.tabs(
    ["📋 Los datos", "📊 Graficos", "🔮 Predecir"]
)


# =====================================================================
# PESTANIA 1: ver los datos
# =====================================================================
with tab_datos:
    st.subheader("Ejemplos del dataset")
    st.write(
        "Esta es una muestra de 20.000 telefonos del dataset original "
        "(el completo tiene 1 millon de filas y pesa demasiado para subirlo)."
    )

    # cuatro numeros resumen, uno al lado del otro
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Telefonos", f"{len(df):,}")
    col2.metric("Marcas", df["brand"].nunique())
    col3.metric("Precio promedio", f"{df['resale_price'].mean():,.0f}")
    col4.metric("Precio maximo", f"{df['resale_price'].max():,.0f}")

    # el usuario elige cuantas filas ver
    cuantas = st.slider("Cuantas filas quieres ver", 5, 50, 10)
    st.dataframe(df.head(cuantas))

    st.subheader("Resumen de las columnas numericas")
    st.dataframe(df.describe().round(2))


# =====================================================================
# PESTANIA 2: graficos
# =====================================================================
with tab_graficos:
    st.subheader("Como se distribuyen los precios")

    # Grafico 1: histograma del precio con la mediana marcada
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["resale_price"], bins=50, color="#2563eb")
    ax.axvline(MEDIANA, color="red", linestyle="--", label=f"Mediana = {MEDIANA:,.0f}")
    ax.set_xlabel("Precio de reventa")
    ax.set_ylabel("Cantidad de telefonos")
    ax.legend()
    st.pyplot(fig)
    st.caption("Los telefonos a la derecha de la linea roja son los de 'precio alto'.")

    st.divider()

    # Graficos 2 y 3: uno al lado del otro
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Precio promedio por marca")
        precio_marca = df.groupby("brand")["resale_price"].mean().sort_values()
        st.bar_chart(precio_marca)

    with col_b:
        st.subheader("Precio promedio por condicion")
        orden = ["Poor", "Fair", "Good", "Excellent"]
        precio_cond = df.groupby("condition")["resale_price"].mean().reindex(orden)
        st.bar_chart(precio_cond)

    st.divider()

    # Grafico 4: el usuario elige que variable comparar contra el precio
    st.subheader("Comparar el precio contra otra variable")
    opciones = ["age_months", "battery_health", "original_price",
                "processor_score", "camera_score", "market_demand_score"]
    variable = st.selectbox("Elige una variable", opciones)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    # uso solo 2000 puntos para que el grafico no quede tan cargado
    muestra_grafico = df.sample(2000, random_state=1)
    ax2.scatter(muestra_grafico[variable], muestra_grafico["resale_price"],
                alpha=0.3, s=12, color="#2563eb")
    ax2.set_xlabel(variable)
    ax2.set_ylabel("Precio de reventa")
    st.pyplot(fig2)

    st.divider()

    # Grafico 5: cuantos telefonos hay de cada marca
    st.subheader("Cuantos telefonos hay de cada marca")
    st.bar_chart(df["brand"].value_counts())


# =====================================================================
# PESTANIA 3: predecir
# =====================================================================
with tab_prediccion:
    st.subheader("Ingresa los datos del telefono")
    st.write("Cambia los valores y presiona el boton para ver la prediccion.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Ficha tecnica**")
        release_year = st.number_input("Anio de lanzamiento", 2019, 2025, 2023)
        ram_gb = st.number_input("RAM (GB)", 4, 16, 8)
        storage_gb = st.number_input("Almacenamiento (GB)", 64, 1024, 256)
        screen_size_inches = st.number_input("Pantalla (pulgadas)", 5.5, 7.0, 6.5, 0.1)
        battery_capacity = st.number_input("Bateria (mAh)", 3000, 6500, 4500, 50)
        processor_score = st.slider("Puntaje procesador", 40, 100, 75)
        camera_score = st.slider("Puntaje camara", 40, 100, 80)
        market_demand_score = st.slider("Demanda de mercado", 40, 100, 70)
        original_price = st.number_input("Precio original", 10000, 150000, 60000, 500)

    with col2:
        st.markdown("**Uso y estado**")
        purchase_year = st.number_input("Anio de compra", 2019, 2025, 2023)
        age_months = st.number_input("Edad (meses)", 1, 71, 24)
        usage_hours_per_day = st.number_input("Horas de uso por dia", 1.0, 12.0, 4.0, 0.5)
        battery_health = st.slider("Salud de la bateria (%)", 55, 100, 85)
        warranty_remaining_months = st.number_input("Garantia restante (meses)", 0, 24, 6)
        has_5g = st.checkbox("Tiene 5G", value=True)
        screen_cracked = st.checkbox("Pantalla rota", value=False)
        body_damage = st.checkbox("Danio en el cuerpo", value=False)
        repair_history = st.checkbox("Fue reparado antes", value=False)
        water_damage = st.checkbox("Danio por agua", value=False)

    with col3:
        st.markdown("**Otros datos**")
        brand = st.selectbox("Marca",
                             ["Apple", "Google", "OnePlus", "Realme", "Samsung", "Vivo", "Xiaomi"])
        os_type = st.selectbox("Sistema operativo", ["Android", "iOS"])
        condition = st.selectbox("Condicion", ["Excellent", "Good", "Fair", "Poor"])
        city_tier = st.selectbox("Nivel de ciudad", ["Tier1", "Tier2", "Tier3"])
        seller_type = st.selectbox("Tipo de vendedor", ["Individual", "Store"])
        box_available = st.checkbox("Tiene caja", value=True)
        charger_available = st.checkbox("Tiene cargador", value=True)

    # --- Boton de prediccion ---
    if st.button("Predecir precio", type="primary"):
        # 1) armo un diccionario con todo lo que ingreso el usuario
        #    (los checkbox devuelven True/False, los paso a 1/0 con int())
        datos = {
            "release_year": release_year,
            "ram_gb": ram_gb,
            "storage_gb": storage_gb,
            "screen_size_inches": screen_size_inches,
            "battery_capacity": battery_capacity,
            "processor_score": processor_score,
            "camera_score": camera_score,
            "has_5g": int(has_5g),
            "original_price": original_price,
            "purchase_year": purchase_year,
            "age_months": age_months,
            "usage_hours_per_day": usage_hours_per_day,
            "battery_health": battery_health,
            "screen_cracked": int(screen_cracked),
            "body_damage": int(body_damage),
            "repair_history": int(repair_history),
            "water_damage": int(water_damage),
            "warranty_remaining_months": warranty_remaining_months,
            "box_available": int(box_available),
            "charger_available": int(charger_available),
            "market_demand_score": market_demand_score,
            "brand": brand,
            "os_type": os_type,
            "condition": condition,
            "city_tier": city_tier,
            "seller_type": seller_type,
        }

        # 2) lo convierto en una fila de DataFrame
        fila = pd.DataFrame([datos])

        # 3) aplico get_dummies igual que en el entrenamiento
        fila = pd.get_dummies(fila, columns=CATEGORICAS)

        # 4) dejo exactamente las columnas del modelo (las que falten van en 0)
        fila = fila.reindex(columns=COLUMNAS, fill_value=0)

        # 5) predigo
        proba_alto = modelo.predict_proba(fila)[0, 1]
        clase = 1 if proba_alto >= 0.5 else 0

        st.divider()
        if clase == 1:
            st.success(f"### {CLASES[1]}")
        else:
            st.error(f"### {CLASES[0]}")

        st.metric("Probabilidad de precio alto", f"{proba_alto * 100:.1f}%")
        st.progress(float(proba_alto))
        st.caption(f"'Alto' significa por encima de la mediana ({MEDIANA:,.2f}).")
