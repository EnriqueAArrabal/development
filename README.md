# LVM Autonomous Driving

Proyecto de investigación y experimentación con **Large Vision Models (LVMs)** aplicados a conducción autónoma, incluyendo detección de objetos, reconocimiento visual y generación de descripciones.

## 📋 Descripción

Este repositorio contiene implementaciones y comparativas de diferentes modelos de visión de gran escala para aplicaciones de conducción autónoma:

- **YOLOv11**: Detección de objetos en tiempo real
- **GroundingDINO**: Detección de objetos con lenguaje natural
- **LLaVa**: Modelo multimodal para comprensión visual y generación de texto

## 🚀 Características

- ✅ Implementación de múltiples modelos de visión (YOLO, GroundingDINO, LLaVa)
- ✅ Análisis de métricas de rendimiento
- ✅ Evaluación de uso de memoria
- ✅ Medición de tiempos de inferencia
- ✅ Comparación de tamaños de modelos
- ✅ Detección de objetos y generación de descripciones de imágenes

## 📁 Estructura del Proyecto

```
lvm-autonomous-driving/
├── image_captioning/         # Módulo de generación de descripciones
├── object_detection/          # Módulo de detección de objetos
├── results/                   # Resultados de experimentos y visualizaciones
├── scripts_memory/            # Scripts para análisis de uso de memoria
├── scripts_metrics/           # Scripts para evaluación de métricas
├── scripts_sizes/             # Scripts para análisis de tamaños de modelos
├── scripts_time_inference/    # Scripts para medición de tiempos de inferencia
├── tools/                     # Utilidades y herramientas auxiliares
├── GroundingDINO_example.py   # Ejemplo de uso de GroundingDINO
├── LLaVa_example.py          # Ejemplo de uso de LLaVa
├── YOLOv11_example.py        # Ejemplo de uso de YOLOv11
├── yolo11n.pt                # Modelo YOLOv11 nano pre-entrenado
└── .gitignore
```

## 🛠️ Tecnologías

- **Python 3.8+**
- **PyTorch** / **TensorFlow**
- **YOLOv11** (Ultralytics)
- **GroundingDINO**
- **LLaVa** (Large Language and Vision Assistant)
- **OpenCV**
- **NumPy**, **Matplotlib**

## 📦 Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/EnriqueAArrabal/lvm-autonomous-driving.git
cd lvm-autonomous-driving
```

2. Crea un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install numpy matplotlib
# Instalar GroundingDINO y LLaVa según sus respectivas instrucciones
```

## 💻 Uso

### Detección de objetos con YOLOv11
```bash
python YOLOv11_example.py
```

### Detección con lenguaje natural usando GroundingDINO
```bash
python GroundingDINO_example.py
```

### Generación de descripciones con LLaVa
```bash
python LLaVa_example.py
```

### Ejecutar análisis de rendimiento
```bash
# Métricas de rendimiento
python scripts_metrics/[script_name].py

# Análisis de memoria
python scripts_memory/[script_name].py

# Medición de tiempos
python scripts_time_inference/[script_name].py
```

## 📊 Experimentos

El proyecto incluye scripts para evaluar diferentes aspectos de los modelos:

- **Métricas**: Precisión, recall, mAP, etc.
- **Memoria**: Uso de RAM y VRAM durante la inferencia
- **Tiempo**: Latencia y throughput de cada modelo
- **Tamaño**: Comparación de tamaños de modelos

Los resultados se almacenan en la carpeta `results/`.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Haz fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/NuevaCaracteristica`)
3. Commit tus cambios (`git commit -m 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/NuevaCaracteristica`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está disponible para uso educativo y de investigación.

## 👤 Autor

**Enrique A. Arrabal**
- GitHub: [@EnriqueAArrabal](https://github.com/EnriqueAArrabal)

## 📧 Contacto

Si tienes preguntas o sugerencias, no dudes en abrir un issue en el repositorio.

---

⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!
