# 📱 Resumen Completo: Sistema de Detección de Equipo Militar

## 🎯 Objetivo
Desarrollar una aplicación móvil Android que detecte en tiempo real si un soldado tiene todo el equipo de seguridad requerido (7 elementos) usando Inteligencia Artificial y visión por computadora.

---

## 🔧 Herramientas Utilizadas

### 1. **Roboflow** - Gestión y Anotación de Dataset
**Propósito:** Crear, anotar y versionar el dataset de imágenes
- **Versiones:**
  - v1: ~1,700 imágenes anotadas (7 clases)
  - v2: ~2,000+ imágenes anotadas (expansión del dataset)
- **Clases anotadas:** armaP, botas, buff, casco, chaleco, gafas, uniforme
- **Formato de salida:** Dataset en formato YOLO v8

---

### 2. **YOLOv8n (You Only Look Once)** - Modelo de Detección
**Propósito:** Red neuronal pre-entrenada para detección de objetos
- **Modelo base:** `yolov8n.pt` (nano - ultraligero, ~3.2MB)
- **Entrenamiento:**
  - Epochs: 50
  - Batch size: 16
  - Image size: 416x416
  - Resultados: 92.6% mAP50
- **Dispositivo:** CPU (Windows)

**Archivo:** `C:\Users\ronny\Downloads\Modelo Entrenado\best.pt`

---

### 3. **Python 3.10** - Backend de Inferencia
**Propósito:** Procesar imágenes y ejecutar el modelo

**Librerías principales:**
```python
ultralytics==8.0.0        # YOLO
flask==2.3.0              # Servidor web
opencv-python==4.8.0      # Procesamiento de imágenes
numpy==1.24.0             # Cálculos numéricos
Pillow==9.5.0             # Decodificación de imágenes
```

**Archivo clave:** `flask_api.py`

---

### 4. **Flask** - Servidor API REST
**Propósito:** Exponer el modelo como servicio HTTP

**Endpoints:**
- `POST /detect` - Recibe imagen base64, retorna detecciones
- `GET /health` - Verifica estado del servidor

**Configuración:**
```python
Host: 0.0.0.0
Puerto: 5000
Protocolo: HTTP
```

---

### 5. **ngrok** - Túnel Público
**Propósito:** Exponer el servidor local a Internet para que amigos en otra ciudad accedan

**Configuración:**
```
Comando: ngrok http 5000
URL pública: https://[ID].ngrok-free.app -> http://localhost:5000
Región: South America (sa)
```

---

### 6. **Flutter/Dart** - Aplicación Móvil
**Propósito:** Interfaz de usuario para Android

**Dependencias:**
```yaml
camera: ^0.10.5              # Acceso a cámara
permission_handler: ^12.0.3  # Solicitar permisos
http: ^1.1.0                 # Llamadas REST
image: ^4.1.5                # Procesamiento de imágenes
path_provider: ^2.1.0        # Acceso a archivos
archive: ^3.4.0              # Manipulación de archivos
```

**Pantallas:**
1. **LoginScreen** - Autenticación (Usuario: Yair, Contraseña: 2026)
2. **CameraScreen** - Captura de fotos
3. **ResultsScreen** - Visualización de resultados

---

### 7. **Android SDK** - Compilación a APK
**Propósito:** Generar ejecutable para dispositivos Android

**Archivo final:**
```
C:\Users\ronny\OneDrive\Documents\Copia de seguridad\Copia\Codigo Deteccion\codigo_deteccion\build\app\outputs\flutter-apk\app-release.apk
```

**Permisos configurados:**
- `android.permission.CAMERA` - Acceso a cámara
- `android.permission.INTERNET` - Conexión a servidor

---

### 8. **Git/GitHub** - Control de Versiones
**Propósito:** Versionado del código fuente

**Repositorio:** ProyectoDeteccionObjetos

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      USUARIO (Android)                       │
│                    Smartphone/Tablet                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ HTTP + ngrok
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   ngrok Public Tunnel                        │
│            https://[ID].ngrok-free.app                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ HTTP Localhost
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   Flask API Server                           │
│              (Windows Local Machine)                         │
│                  http://0.0.0.0:5000                         │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Pipeline de Detección Inteligente          │   │
│  │                                                      │   │
│  │ 1. Recibir imagen base64                            │   │
│  │ 2. Decodificar (cv2 o PIL)                          │   │
│  │ 3. Detectar personas (yolov8n genérico)             │   │
│  │ 4. Aplicar recortes estratégicos:                    │   │
│  │    - Imagen completa                                │   │
│  │    - Mitad superior (cabeza/torso)                  │   │
│  │    - Mitad inferior (botas/armas)                   │   │
│  │ 5. Ejecutar modelo especializado (best.pt)          │   │
│  │ 6. Aplicar umbrales inteligentes por clase:         │   │
│  │    - Gafas: 65%                                      │   │
│  │    - Chaleco/Botas/Casco: 40%                       │   │
│  │    - Arma/Uniforme/Buff: 30%                        │   │
│  │ 7. Análisis HSV para uniforme (color verde/caqui)   │   │
│  │ 8. Consolidar resultados                            │   │
│  │ 9. Retornar JSON con detecciones                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│            [best.pt: 92.6% mAP50]                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Flujo de Detección Completo

### **Paso 1: Captura de Imagen**
```
Usuario abre app → Login (Yair/2026) → Ve cámara → Captura foto
```

### **Paso 2: Envío a Servidor**
```
Imagen JPG → Base64 encoding → POST /detect → Flask recibe
```

### **Paso 3: Pipeline Dual-Model**
```
Imagen → yolov8n.pt (detectar personas) → Aislar cada soldado
```

### **Paso 4: Recortes Estratégicos**
```
Para cada persona:
  ├─ Crop full (100%) → best.pt
  ├─ Crop superior (75%) → best.pt (cabeza/buff/chaleco)
  └─ Crop inferior (70%) → best.pt (botas/arma)
```

### **Paso 5: Umbrales Inteligentes**
```
Detectar clase → Verificar confianza ≥ threshold de esa clase
  ├─ Gafas: ≥ 65%
  ├─ Chaleco/Botas: ≥ 40%
  └─ Arma/Uniforme: ≥ 30%
```

### **Paso 6: Análisis HSV (Uniforme)**
```
Si uniforme no detectó:
  → Convertir a HSV
  → Buscar píxeles verdes/caqui
  → Si > 15% → Uniforme detectado
```

### **Paso 7: Consolidación**
```
Todos los detectados → Comparar vs. REQUIRED (7 elementos)
  ├─ Si TODOS: APTO ✅
  └─ Si FALTA ALGUNO: NO APTO ❌ + listar faltantes
```

### **Paso 8: Respuesta JSON**
```json
{
  "apto": true/false,
  "detected": {
    "buff": 0.66,
    "casco": 0.87,
    "chaleco": 0.59,
    "gafas": 0.73,
    "uniforme": 0.85
  },
  "missing": ["botas", "armaP"],
  "message": "APTO ✅ / NO APTO ❌"
}
```

### **Paso 9: Visualización**
```
App recibe JSON → Renderiza ResultsScreen
  ├─ Imagen capturada
  ├─ Estado APTO/NO APTO (verde/rojo)
  ├─ Equipos detectados con confianza
  ├─ Equipos faltantes
  └─ Botones: Volver a Capturar / Salir
```

---

## 🔑 Configuraciones Clave

### **Umbrales de Confianza por Clase**
```python
CLASS_CONFIDENCES = {
    "gafas": 0.65,      # Alto - evita falsos positivos
    "chaleco": 0.40,
    "botas": 0.40,
    "casco": 0.25,      # Bajo - difícil de detectar
    "armaP": 0.30,      # Bajo - se deforma/oculta
    "uniforme": 0.30,   # Bajo - análisis HSV como backup
    "buff": 0.35,
}
```

### **Rango HSV para Uniforme**
```python
# Verde militar
lower_green1 = np.array([35, 30, 30])
upper_green1 = np.array([90, 255, 255])

# Marrón/Caqui
lower_brown = np.array([10, 30, 30])
upper_brown = np.array([25, 255, 255])

# Umbral: > 15% de píxeles = Uniforme detectado
```

---

## 📱 Credenciales de Acceso

| Campo | Valor |
|-------|-------|
| Usuario | Yair |
| Contraseña | 2026 |

---

## 🚀 Cómo Usar la App

### **1. Instalación**
```bash
# Copiar APK a dispositivo Android
adb install app-release.apk
```

### **2. Primer uso**
```
Abrir app → Ingresar credenciales (Yair/2026)
→ Permitir acceso a cámara → Ver cámara en tiempo real
```

### **3. Capturar foto**
```
Acercarse a la persona → Pulsar botón circular de cámara
→ Esperar procesamiento (~2-3 segundos)
```

### **4. Ver resultados**
```
Pantalla de resultados mostrará:
  - APTO ✅ (tiene todos los 7 equipos)
  - NO APTO ❌ (falta alguno) + equipos faltantes
```

### **5. Volver a capturar**
```
Pulsar "Volver a Capturar" → Repetir desde paso 3
```

---

## 💾 Archivos Generados

| Ruta | Descripción |
|------|-------------|
| `flask_api.py` | Servidor con pipeline completo |
| `equipment_classifier.dart` | Cliente HTTP en Flutter |
| `login_screen.dart` | Pantalla de login |
| `camera_screen.dart` | Captura de fotos |
| `results_screen.dart` | Visualización de resultados |
| `app-release.apk` | Aplicación compilada para Android |

---

## 🔌 URLs de Conexión

| Componente | URL |
|------------|-----|
| Flask Local | http://192.168.0.109:5000 |
| Flask Alternativo | http://127.0.0.1:5000 |
| ngrok Public | https://[ID].ngrok-free.app |

---

## 📈 Métricas del Modelo

| Métrica | Valor |
|---------|-------|
| Modelo | YOLOv8n (nano) |
| Tamaño | ~3.2 MB |
| mAP50 | 92.6% |
| Clases | 7 (armaP, botas, buff, casco, chaleco, gafas, uniforme) |
| Imágenes entrenamiento | 2,000+ |
| Epochs | 50 |
| Batch size | 16 |
| Image size | 416x416 |

---

## 🎨 Interfaz de Usuario

### **Pantalla de Login**
- Fondo azul degradado
- Campo Usuario + Contraseña
- Botón "INICIAR SESIÓN" (56px altura)
- Solicita permiso de cámara

### **Pantalla de Cámara**
- Vista en vivo de la cámara
- Mensaje: "📷 Captura una foto para detectar equipo"
- Botón circular azul para capturar

### **Pantalla de Resultados**
- Imagen capturada
- Estado grande (APTO ✅ / NO APTO ❌)
- Lista de equipos detectados (verde)
- Lista de equipos faltantes (rojo)
- Botones: "Volver a Capturar" / "Salir"

---

## 🔐 Seguridad

| Aspecto | Implementación |
|--------|-----------------|
| Autenticación | Usuario/Contraseña hardcoded (Yair/2026) |
| HTTPS | ngrok proporciona SSL/TLS |
| Permisos Android | Camera + Internet explícitos |
| Validación | Datos JSON validados en servidor |

---

## ⚡ Rendimiento

| Operación | Tiempo Estimado |
|-----------|-----------------|
| Captura de foto | Instantáneo |
| Envío a servidor | 1-2 segundos (red) |
| Procesamiento (pipeline) | 2-3 segundos (CPU) |
| Retorno de resultados | 1 segundo |
| **Total** | **4-6 segundos** |

---

## 🛠️ Solución de Problemas

### **Problema: No detecta uniforme**
**Solución:** El análisis HSV busca colores verde/caqui. Asegurar buena iluminación.

### **Problema: No detecta casco**
**Solución:** Acercarse más a la cámara. El modelo fue entrenado a cierta distancia.

### **Problema: Error de conexión**
**Solución:** Verificar que Flask esté corriendo y ngrok esté activo.

### **Problema: Credenciales incorrectas**
**Usuario:** Yair
**Contraseña:** 2026

---

## 📝 Notas Finales

✅ **App completamente funcional y lista para entregar**
✅ **Pipeline optimizado con recortes estratégicos y análisis HSV**
✅ **Funciona localmente y desde otra ciudad (ngrok)**
✅ **Interfaz amigable y validaciones de seguridad**
✅ **Modelo entrenado a 92.6% mAP50**

---

**Fecha:** 16 de Agosto, 2026
**Usuario:** Joel
**Proyecto:** Sistema de Detección de Equipo Militar
