def read_file_bytes(file_path: str) -> bytes:
    with open(file_path, "rb") as f:
        file_byte = f.read()
    return file_byte


def make_file_form_data_object(file_name, file_content):
    return [("file", (file_name, file_content))]
