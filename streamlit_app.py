import streamlit as st
import numpy as np
from neuron import Neuron
# Mostrar la imagen
st.image("./img/neurona.jpg")

# Título de la aplicación
st.header("Simulador de neurona")

# Diccionario de funciones de activación
func_map = {
    "Sigmoid": "sigmoid",
    "ReLU": "relu",
    "Tangente hiperbólica": "tanh",
    #"Linear": "linear",
    "Binary Step": "binary_step"
}
# Configuración del número de entradas/pesos
n = st.slider("Elige el número de entradas/pesos que tendrá la neurona", min_value=1, max_value=10, value=1)
x = np.zeros(n)
w = np.zeros(n)

# Configuración para los pesos
st.subheader("Pesos")
col_pesos = st.columns(n)  # Crear columnas para los pesos
for i in range(n):
    with col_pesos[i]:
        st.markdown(f"Peso w<sub>{i}</sub>", unsafe_allow_html=True)
        w[i] = st.number_input(f"Peso w<sub>{i}</sub>", value=0.0, key=f"peso_w{i}", label_visibility="collapsed")
st.write(f"w = {w.tolist()}")

# Configuración para las entradas
st.subheader("Entradas")
col_entradas = st.columns(n)  # Crear columnas para las entradas
for i in range(n):
    with col_entradas[i]:
        st.markdown(f"Entrada x<sub>{i}</sub>", unsafe_allow_html=True)
        x[i] = st.number_input(f"Entrada x<sub>{i}</sub>", value=0.0, key=f"entrada_x{i}", label_visibility="collapsed")
st.write(f"x = {x.tolist()}")

col2 = st.columns(2)
with col2[0]:
    # Configuración para el sesgo
    st.subheader("Sesgos")
    bias = st.number_input("Introduce el valor del sesgo", value=0.0, key="bias")
with col2[1]:
    st.subheader("Función de activación")
    func = st.selectbox("Elige la función de activación", list(func_map.keys()), key="func")

# Calcular salida
if st.button("Calcular la salida"):
    n1 = Neuron(weights=w, bias=bias, func=func_map[func])
    salida = n1.run(input_data=x)
    st.write(f"La salida de la neurona es: {salida}")
