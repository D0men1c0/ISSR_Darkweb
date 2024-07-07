import gdown
import os

def download_file_from_google_drive(file_id, destination):
    # Construct the download URL for the file from its ID
    url = f"https://drive.google.com/uc?id={file_id}"
    # Use gdown to download the file to the specified destination
    gdown.download(url, destination, quiet=False)

# List of files to download with their respective Google Drive file IDs and destination paths
files_to_download = [
    {
        "file_id": "1U5a_WuGLzMynoB7a0UaiUU6mZ6zUneii",
        "destination": "prova/prova.parquet"  # Example destination path
    }
]

# Download each file listed in files_to_download
for file in files_to_download:

    # Create the directory structure if it doesn't exist
    os.makedirs(os.path.dirname(file["destination"]), exist_ok=True)

    # Download the file using the defined function
    download_file_from_google_drive(file["file_id"], file["destination"])
    
    # Print a confirmation message after successful download
    print(f"Download completed. The file has been saved to: {file['destination']}")
