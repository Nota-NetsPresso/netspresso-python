from typing import List

from app.api.v1.schemas.system import LibraryInfo
from netspresso.clients.utils.system import get_package_version


class SystemService:
    def get_installed_libraries(self) -> List[LibraryInfo]:
        LIBRARY_KEYS = ["netspresso"]

        installed_libraries = [
            LibraryInfo(name=library_name, version=get_package_version(package_name=library_name))
            for library_name in LIBRARY_KEYS
        ]

        return installed_libraries


system_service = SystemService()
