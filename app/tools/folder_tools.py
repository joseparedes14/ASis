"""
Folder monitoring and ASIORGA tools for the agent.

Provides tools for:
- Managing monitored folders (add, remove, list)
- Managing destination folders (create, delete, list)
- Listing folder contents
"""

from langchain_core.tools import tool

from app.config.logging_config import get_logger
from app.services.folder_manager import FolderManager
from app.services.folder_monitor import FolderMonitor

logger = get_logger(__name__)

# Shared instances (initialized on first use)
_folder_manager: FolderManager | None = None
_folder_monitor: FolderMonitor | None = None


def _get_folder_manager() -> FolderManager:
    global _folder_manager
    if _folder_manager is None:
        _folder_manager = FolderManager()
    return _folder_manager


def _get_folder_monitor() -> FolderMonitor:
    global _folder_monitor
    if _folder_monitor is None:
        _folder_monitor = FolderMonitor()
    return _folder_monitor


@tool
def add_monitored_folder(path: str) -> str:
    """Añade una carpeta al monitoreo. Cuando aparezcan archivos nuevos en ella, \
se procesarán automáticamente y se clasificarán en ASIORGA.

    Soporta rutas absolutas (ej: C:\\Users\\josem\\Downloads) o nombres comunes \
(ej: "Descargas", "Documentos", "Escritorio").

    Args:
        path: Ruta de la carpeta o nombre conocido (ej: "Descargas" o "C:\\Users\\josem\\Downloads").

    Returns:
        Mensaje de confirmación o error.
    """
    monitor = _get_folder_monitor()
    return monitor.add_folder(path)


@tool
def remove_monitored_folder(path: str) -> str:
    """Elimina una carpeta del monitoreo. Los archivos que ya están en ella \
no se seguirán procesando.

    Args:
        path: Ruta de la carpeta o nombre conocido.

    Returns:
        Mensaje de confirmación o error.
    """
    monitor = _get_folder_monitor()
    return monitor.remove_folder(path)


@tool
def list_monitored_folders() -> str:
    """Lista todas las carpetas que están siendo monitoreadas actualmente.

    Returns:
        Lista de carpetas monitoreadas.
    """
    monitor = _get_folder_monitor()
    folders = monitor.list_folders()

    if not folders:
        return "No hay carpetas configuradas para monitoreo."

    lines = ["Carpetas monitoreadas:"]
    for f in folders:
        lines.append(f"  - {f}")
    return "\n".join(lines)


@tool
def create_destination_folder(name: str, description: str) -> str:
    """Crea una nueva carpeta destino dentro de ASIORGA para clasificar documentos.

    Cada carpeta destino tiene una descripción que ayuda al agente a decidir \
dónde guardar cada archivo.

    Args:
        name: Nombre de la carpeta (ej: "Facturas", "Proyectos").
        description: Descripción del tipo de documentos que irán aquí \
(ej: "Facturas, recibos y tickets de compra").

    Returns:
        Mensaje de confirmación o error.
    """
    fm = _get_folder_manager()
    return fm.create_destination(name, description)


@tool
def delete_destination_folder(name: str) -> str:
    """Elimina una carpeta destino de la configuración de ASIORGA.

    La carpeta física se conserva por seguridad, solo se elimina \
la configuración de clasificación.

    Args:
        name: Nombre de la carpeta a eliminar.

    Returns:
        Mensaje de confirmación o error.
    """
    fm = _get_folder_manager()
    return fm.delete_destination(name)


@tool
def list_destination_folders() -> str:
    """Lista todas las carpetas destino configuradas en ASIORGA con sus descripciones.

    Returns:
        Lista de carpetas destino y sus descripciones.
    """
    fm = _get_folder_manager()
    return fm.get_destination_descriptions()


@tool
def list_folder_contents(folder_name: str = "") -> str:
    """Lista el contenido de una carpeta dentro de ASIORGA.

    Si no se especifica nombre, lista el contenido de la raíz de ASIORGA.

    Args:
        folder_name: Nombre de la subcarpeta (ej: "Facturas", "Fotos"). \
Si está vacío, lista la raíz de ASIORGA.

    Returns:
        Contenido de la carpeta con archivos y subcarpetas.
    """
    fm = _get_folder_manager()
    return fm.list_folder_contents(folder_name if folder_name else None)


# Export all folder tools
FOLDER_TOOLS = [
    add_monitored_folder,
    remove_monitored_folder,
    list_monitored_folders,
    create_destination_folder,
    delete_destination_folder,
    list_destination_folders,
    list_folder_contents,
]
