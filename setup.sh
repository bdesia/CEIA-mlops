#!/bin/bash
set -e  # Exit si algo falla

POETRY_PYPROJECT="pyproject.toml"
PROJECT_NAME=$(grep 'name =' $POETRY_PYPROJECT | awk -F '"' '{print $2}' | head -1)

# Validar que se haya extraído el nombre del proyecto
if [ -z "$PROJECT_NAME" ]; then
    echo "Error: No se pudo extraer el nombre del proyecto de $POETRY_PYPROJECT"
    exit 1
fi
echo "Nombre del proyecto: $PROJECT_NAME"

PYTHON_VERSION="3.11"

echo "Configurando entorno con Poetry..."

# Crea carpeta en directorio del repo para guardar el entorno virtual
poetry config virtualenvs.in-project true --local

# Limpia caché antes de crear el entorno
poetry cache clear pypi --all -q

# Borra .venv anterior para evitar bugs si existe un entorno virtual roto
rm -rf .venv

# Usa o crea un virtualenv con la versión específica de Python
poetry env use $PYTHON_VERSION

# Instala todas las dependencias del proyecto
poetry install

# Instala las dependencias principales del proyecto
poetry install --only main
poetry sync

# Instala herramientas de desarrollo
poetry run pip install --quiet ipykernel jupyterlab-widgets==3.0.13 widgetsnbextension

# Registra el kernel en Jupyter
poetry run python -m ipykernel install --user --name=$PROJECT_NAME --display-name "Python ($PROJECT_NAME)"

poetry add torch==2.6.0+cpu --source pytorch-cpu

# Ahora, obtiene la ruta del entorno virtual precisamente donde está en realidad
VENV_PATH=$(poetry env info --path)

# La carpeta site-packages en ese entorno
SITE_PACKAGES="$VENV_PATH/Lib/site-packages"

# Crear la carpeta si no existe
mkdir -p "$SITE_PACKAGES"

# Ruta actual en formato Windows
ABS_PATH=$(cygpath -w "$(pwd)")

# Crea el archivo .pth en la carpeta correcta del entorno virtual
echo "$ABS_PATH\src" > "$SITE_PACKAGES/$PROJECT_NAME.pth"

echo "Archivo .pth creado en: $SITE_PACKAGES/$PROJECT_NAME.pth"
echo "Contenido:"
cat "$SITE_PACKAGES/$PROJECT_NAME.pth"

echo "El entorno con Poetry está listo y configurado."
