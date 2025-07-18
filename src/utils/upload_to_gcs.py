import os
from google.cloud import storage
from pathlib import Path
import datetime

def upload_folder_to_gcs(local_folder_path, bucket_name, destination_prefix=""):
    """
    Upload a local folder to Google Cloud Storage bucket.
    
    Args:
        local_folder_path (str): Path to the local folder to upload
        bucket_name (str): Name of the GCS bucket
        destination_prefix (str, optional): Prefix for the destination path in bucket
    
    Returns:
        list: List of uploaded blob names
    """
    # Initialize the GCS client
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    destination_prefix = f"{destination_prefix}_{timestamp}"
    
    uploaded_files = []
    local_path = Path(local_folder_path)
    
    # Check if local folder exists
    if not local_path.exists():
        raise FileNotFoundError(f"Local folder '{local_folder_path}' does not exist")
    
    if not local_path.is_dir():
        raise NotADirectoryError(f"'{local_folder_path}' is not a directory")
    
    # Walk through all files in the folder recursively
    for file_path in local_path.rglob('*'):
        if file_path.is_file():
            # Calculate relative path from the base folder
            relative_path = file_path.relative_to(local_path)
            
            # Create destination path in bucket
            if destination_prefix:
                destination_path = f"{destination_prefix.rstrip('/')}/{relative_path}"
            else:
                destination_path = str(relative_path)
            
            # Convert Windows path separators to forward slashes for GCS
            destination_path = destination_path.replace('\\', '/')
            
            # Upload the file
            blob = bucket.blob(destination_path)
            blob.upload_from_filename(str(file_path))
            
            uploaded_files.append(destination_path)
            print(f"Uploaded: {file_path} -> gs://{bucket_name}/{destination_path}")
    
    return uploaded_files


