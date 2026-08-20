# Usar imagen base de PyTorch (ya tiene dependencias)
FROM pytorch/pytorch:2.0.1-runtime-ubuntu22.04

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias adicionales
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY flask_api.py .
COPY best.pt .

# Establecer variable de entorno PORT
ENV PORT=8080

# Exponer puerto 8080
EXPOSE 8080

# Comando para ejecutar la aplicación con Gunicorn
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 300 --access-logfile - --error-logfile - flask_api:app
