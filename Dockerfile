FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crear carpeta de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY requirements .
RUN pip install --no-cache-dir -r requirements

COPY app.py .

# Comando para correr la app
CMD ["python", "app.py"]
