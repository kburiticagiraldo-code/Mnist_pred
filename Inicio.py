import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras

st.set_page_config(
    page_title="Ecuaciones a Mano",
    page_icon="🧮",
    layout="wide"
)

# Mostrar versión en sidebar
with st.sidebar:
    st.write(f"**TensorFlow:** {tf.__version__}")
    st.write(f"**Keras:** {keras.__version__}")
    st.write("**Consejos**")
    st.write("- Escribe grande y con trazo grueso")
    st.write("- Deja un espacio entre dígitos (que no se toquen)")
    st.write("- Solo números enteros: la operación se elige aparte")

st.title("🧮 Ecuaciones Escritas a Mano")
st.markdown("**Modelo CNN entrenado en Google Colab con MNIST** · escribe dos números, elige la operación y se resuelve")

# Cargar modelo
@st.cache_resource
def load_mnist_model():
    """Carga el modelo MNIST desde archivo .keras"""

    # Buscar archivo del modelo
    model_files = ['mnist_model.keras', 'mnist_model.h5']

    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                st.info(f"📂 Intentando cargar: {model_path}")
                model = keras.models.load_model(model_path, compile=False)
                st.success(f"✅ Modelo cargado exitosamente desde: `{model_path}`")
                return model
            except Exception as e:
                st.error(f"❌ Error al cargar {model_path}: {str(e)}")
                continue

    # Si no se encuentra el modelo
    st.error("❌ No se encontró ningún modelo")
    st.info("""
    **Instrucciones:**
    1. Entrena el modelo en Google Colab
    2. Descarga `mnist_model.keras`
    3. Súbelo a la raíz de tu repositorio
    4. Reinicia la aplicación
    """)
    st.stop()

# Cargar modelo
model = load_mnist_model()


# ---------- Lectura de números (varios dígitos) ----------
def extraer_digitos(gris):
    """Separa los dígitos de izquierda a derecha y los deja en formato MNIST (28x28)."""
    tinta = gris > 50
    columnas = tinta.any(axis=0)

    # Un dígito = un grupo seguido de columnas con tinta
    segmentos, inicio = [], None
    for x, hay in enumerate(columnas):
        if hay and inicio is None:
            inicio = x
        elif not hay and inicio is not None:
            segmentos.append((inicio, x))
            inicio = None
    if inicio is not None:
        segmentos.append((inicio, len(columnas)))

    digitos = []
    for x0, x1 in segmentos:
        filas = np.where(tinta[:, x0:x1].any(axis=1))[0]
        y0, y1 = filas[0], filas[-1] + 1
        recorte = gris[y0:y1, x0:x1]
        alto, ancho = recorte.shape
        if max(alto, ancho) < 10:  # ruido
            continue

        # Escalar a 20 px conservando proporción y centrar en 28x28, como MNIST
        escala = 20.0 / max(alto, ancho)
        nuevo = (max(1, round(ancho * escala)), max(1, round(alto * escala)))
        im = Image.fromarray(recorte).resize(nuevo, Image.Resampling.LANCZOS)
        lienzo = Image.new("L", (28, 28), 0)
        lienzo.paste(im, ((28 - nuevo[0]) // 2, (28 - nuevo[1]) // 2))
        digitos.append(np.array(lienzo) / 255.0)
    return digitos


def leer_numero(image_data):
    """Devuelve (numero, imagenes, digitos, confianzas) o None si no hay nada escrito."""
    if image_data is None:
        return None
    gris = np.array(Image.fromarray(image_data.astype("uint8")).convert("L"))
    digitos = extraer_digitos(gris)
    if not digitos:
        return None

    prediction = model.predict(np.array(digitos).reshape(-1, 28, 28, 1), verbose=0)
    digs = prediction.argmax(axis=1)
    confs = prediction.max(axis=1) * 100
    numero = int("".join(str(d) for d in digs))
    return numero, digitos, digs, confs


def mostrar_digitos(titulo, imgs, digs, confs):
    st.write(f"**{titulo}:**")
    st.image(
        [(i * 255).astype("uint8") for i in imgs],
        caption=[f"{d} ({c:.0f}%)" for d, c in zip(digs, confs)],
        width=70,
    )


# ---------- Interfaz ----------
st.write("### 🎨 Escribe los dos números y elige la operación")

# El contador permite limpiar los paneles (cambia la clave del canvas)
if "n" not in st.session_state:
    st.session_state["n"] = 0
n = st.session_state["n"]

col1, col2, col3 = st.columns([4, 1.3, 4])

with col1:
    st.write("**Número 1**")
    canvas_a = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        width=400,
        height=200,
        drawing_mode="freedraw",
        key=f"canvas_a_{n}",
    )

with col2:
    st.write("**Operación**")
    op = st.selectbox("Operación", ["+", "−", "×", "÷"], label_visibility="collapsed")

with col3:
    st.write("**Número 2**")
    canvas_b = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        width=400,
        height=200,
        drawing_mode="freedraw",
        key=f"canvas_b_{n}",
    )

btn1, btn2 = st.columns(2)
with btn1:
    predict_btn = st.button("🔍 **Resolver**", type="primary", use_container_width=True)
with btn2:
    if st.button("🗑️ Limpiar", use_container_width=True):
        st.session_state["n"] += 1
        st.rerun()

# ---------- Resolución ----------
if predict_btn:
    with st.spinner("🤔 Leyendo los números..."):
        a = leer_numero(canvas_a.image_data)
        b = leer_numero(canvas_b.image_data)

    if a is None or b is None:
        st.warning("⚠️ Por favor, escribe un número en cada panel")
    else:
        num_a, imgs_a, digs_a, confs_a = a
        num_b, imgs_b, digs_b, confs_b = b

        if op == "÷" and num_b == 0:
            st.error("❌ No se puede dividir entre cero")
        else:
            if op == "+":
                res = num_a + num_b
            elif op == "−":
                res = num_a - num_b
            elif op == "×":
                res = num_a * num_b
            else:
                res = num_a / num_b
                res = int(res) if res.is_integer() else round(res, 4)

            st.success(f"## 🎯 {num_a} {op} {num_b} = **{res}**")

            if min(confs_a.min(), confs_b.min()) < 70:
                st.warning("⚠️ Algún dígito tiene baja confianza. Revisa abajo y escríbelo más grande y claro.")

            st.write("### 🖼️ Así leyó tus dígitos")
            c_a, c_b = st.columns(2)
            with c_a:
                mostrar_digitos("Número 1", imgs_a, digs_a, confs_a)
            with c_b:
                mostrar_digitos("Número 2", imgs_b, digs_b, confs_b)
