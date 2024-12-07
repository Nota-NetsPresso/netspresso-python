from fastapi import status

STATUS_MAP = {
    "PROJECT40001": status.HTTP_400_BAD_REQUEST,
    "PROJECT40901": status.HTTP_409_CONFLICT,
    "PROJECT50001": status.HTTP_500_INTERNAL_SERVER_ERROR,
}