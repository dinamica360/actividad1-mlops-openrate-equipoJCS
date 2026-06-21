# mlops-python-ambiente

Ambiente de pruebas MLOps con Python, Poetry y DVC.

## Requisitos

Antes de empezar, asegúrate de tener instalado:

- Git
- Python 3.13
- Poetry
- DVC
- Plugin SSH de DVC (`dvc-ssh`)

## Clonar el repositorio

```bash
git clone https://github.com/dinamica360/mlops-python-ambiente.git
cd mlops-python-ambiente
```

## Instalar dependencias

```bash
poetry install
```

Si DVC no tiene soporte para SSH en tu entorno, instala el plugin:

```bash
poetry add dvc-ssh
```

## Cómo funciona el dataset

Este repositorio **no guarda el archivo CSV pesado directamente en GitHub**.

- GitHub guarda el código del proyecto.
- GitHub guarda los metadatos de DVC, por ejemplo `data/openrate.csv.dvc`.
- El archivo real del dataset se almacena en un servidor Ubuntu remoto configurado como remote de DVC por SSH.

## Acceso al dataset

Para poder descargar el dataset necesitas tener acceso SSH autorizado al servidor remoto.

Eso significa que:

- Debes tener un usuario habilitado en el servidor Ubuntu, o
- Debes tener tu clave pública SSH registrada en el servidor

Sin ese acceso, `dvc pull` no podrá descargar el archivo.

## Descargar el dataset

Después de clonar el repositorio e instalar dependencias, ejecuta:

```bash
poetry run dvc pull
```

Ese comando descargará los archivos versionados por DVC desde el servidor remoto a tu copia local del proyecto.

## Verificar que el dataset quedó disponible

Puedes comprobarlo con:

```bash
ls -lh data/
```

o abrir directamente el archivo:

```bash
data/openrate.csv
```

## Flujo básico de trabajo

### Obtener últimos cambios

```bash
git pull
poetry run dvc pull
```

### Si actualizas el dataset

```bash
poetry run dvc add data/openrate.csv
git add data/openrate.csv.dvc data/.gitignore
git commit -m "Actualiza versión del dataset"
poetry run dvc push
git push origin main
```

## Estructura esperada

```bash
.
├── .dvc/
├── data/
│   ├── .gitignore
│   ├── openrate.csv.dvc
│   └── processed/
├── notebooks/
├── outputs/
├── src/
├── pyproject.toml
├── poetry.lock
└── README.md
```

## Notas importantes

- No subir `data/openrate.csv` directamente a GitHub.
- El archivo real del dataset debe manejarse con DVC.
- Si cambia el remote o el acceso SSH, hay que actualizar la configuración de DVC.
- Si no tienes permisos al servidor Ubuntu, solicita acceso antes de ejecutar `dvc pull`.

## Validación rápida

Si todo está correcto, este flujo debe funcionar:

```bash
git clone https://github.com/dinamica360/mlops-python-ambiente.git
cd mlops-python-ambiente
poetry install
poetry run dvc pull
ls -lh data/
```

## Soporte

Si `dvc pull` falla, revisar:

- acceso SSH al servidor,
- existencia del plugin `dvc-ssh`,
- configuración del remote en `.dvc/config`.