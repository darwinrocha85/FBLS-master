# FBLS-master

Tesis de grado (Univ. de Carabobo): reconocimiento de emociones multimodal (audio, video,
texto, dataset IEMOCAP) comparando arquitecturas de fusión — incluyendo Fuzzy Broad
Learning System (FBLS). Trabajo previo al repo
[fuzzy-embracenet](https://github.com/darwinrocha85/fuzzy-embracenet), que continúa esta
línea con activación difusa gaussiana.

## Carpetas
| Carpeta | Qué es |
|---|---|
| `MultimodalEmotionFusionUnifyFBLS/` | Variante vigente: fusión unificada + `BLS.py` y `FUZZY-BLS.py` |
| `MultimodalEmotionFusion/` | Base: modelos (Attention, DeepFusion, EmbraceNet, MLP, TensorFusion...), datasets IEMOCAP y resultados |
| `FBLS-python/` | Implementación plana de FBLS con CSVs de entrenamiento |
| `correr proyecto TesisUSB` | Notas para correr el proyecto |

## Dataset
IEMOCAP (solicitar acceso en la web del dataset; no incluido por licencia).

## Nota Windows
Este repo contiene archivos `*:Zone.Identifier` (marcas de descargas de Windows) que impiden
clonarlo completo en Windows (`invalid path`). Clonar con `--no-checkout` o desde Linux/macOS.
