# Proyecto de Estimación de Precio de Carros

Este proyecto tiene como objetivo predecir el precio de un automóvil usado basado en diversas características del vehículo. La predicción se realiza utilizando un modelo de machine learning, y la interfaz está construida con **FastAPI** y **HTML/CSS**.

## Descripción

El usuario puede ingresar detalles sobre el vehículo, como la marca, modelo, edad, kilómetros recorridos, tipo de vendedor, tipo de combustible, entre otros. Al enviar el formulario, el sistema procesará los datos y devolverá una estimación del precio del vehículo basado en el modelo de predicción.

## Tecnologías Utilizadas

- **FastAPI**: Framework para crear la API backend.
- **HTML/CSS**: Para la creación de la interfaz de usuario.
- **Modelo de Machine Learning**: Para la predicción del precio del vehículo.
- **Python**: Lenguaje de programación utilizado para el backend y el modelo de machine learning.

## Características

- Formulario interactivo para ingresar detalles del vehículo.
- Predicción del precio basada en un modelo entrenado.
- Estilo de la interfaz en colores naranja, blanco y negro.
- Responsivo y fácil de usar.

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu_usuario/estimacion-precio-carros.git

Uso
Ejecuta la aplicación FastAPI:

bash
Copiar código
uvicorn main:app --reload
o
python -m uvicorn main:app --reload
Abre tu navegador y accede a http://localhost:8000 para interactuar con el formulario y realizar las predicciones de precio.

Llena los campos requeridos y haz clic en "Estimar Precio" para obtener la predicción del precio del vehículo.


### Explicación de los elementos:

1. **Descripción general del proyecto**: Se detalla el propósito del proyecto, su funcionalidad y cómo puede ser utilizado.
2. **Tecnologías utilizadas**: Enumera las herramientas y lenguajes que se usaron para desarrollar la aplicación.
3. **Instrucciones de instalación**: Explica cómo instalar las dependencias necesarias para ejecutar el proyecto.
4. **Uso del proyecto**: Muestra los pasos para ejecutar la aplicación localmente.
5. **Estructura del proyecto**: Detalla la estructura de carpetas y archivos del proyecto.
6. **Contribuciones y licencia**: Proporciona información sobre cómo contribuir al proyecto y la licencia bajo la cual se distribuye.

Este `README.md` cubre todos los aspectos básicos para ayudar a cualquier persona que quiera colaborar o ejecutar el proyecto. Si deseas hacer cambios adicionales o agregar más detalles, ¡avísame!