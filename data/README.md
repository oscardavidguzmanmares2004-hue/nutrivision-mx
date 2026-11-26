# data/

Esta carpeta **no contiene imágenes** en el repositorio: solo los scripts e instrucciones para obtenerlas.

| Conjunto | Cómo se obtiene | Destino |
|---|---|---|
| Food-101 (8 platillos mexicanos + muestra "otro") | `python data/download.py` (lee el .tar.gz original como stream, ~5 GB de red, ~300 MB en disco) | `data/raw/` |
| Particiones train/val/test sin fuga | `python -m src.make_splits` | `data/splits/*.csv` |
| Fotos propias del equipo (fuera de distribución) | Descargar `ood_equipo.zip` desde la sección **Releases** del repositorio y descomprimir en `data/ood_originales/`; luego `python -m src.prepare_ood` | `data/ood/` |

- Fuente: Bossard, Guillaumin & Van Gool, *Food-101 – Mining Discriminative Components with Random Forests*, ECCV 2014. <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>
- Ficha técnica completa: [`docs/datasheet.md`](../docs/datasheet.md).
- `data/splits/reporte_particiones.json` (sí versionado) documenta conteos por clase y los duplicados eliminados.
