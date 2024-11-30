import zipfile


def extract_data_zip(zip_path, extract_to):
    """
    Unzips the data.zip file if it hasn't been unzipped already.

    Args:
        zip_path (str): Path to the data.zip file.
        extract_to (str): Directory where the contents should be extracted.
    """
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction completed. Files are available in {extract_to}.")
