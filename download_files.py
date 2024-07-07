import os
import gdown
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_file_from_google_drive(file_id, destination):
    # Construct the download URL for the file from its ID
    url = f"https://drive.google.com/uc?id={file_id}"
    # Use gdown to download the file to the specified destination
    gdown.download(url, destination, quiet=False)

# List of files to download with their respective Google Drive file IDs and destination paths
files_to_download = [
    {
        "file_id": "",
        "destination": "Analyze_files/CombiningAnalysisCompleteDataset/ContentAnalysis/DatasetsContentBERTopic/BERTopic_all-MiniLM-L6-v2_190_20n_8dim_prova.parquet"
    }
]

# Download each file listed in files_to_download
for file in tqdm(files_to_download, desc="Downloading files"):
    while True:
        should_download = input(f"Do you want to download the file to {file['destination']}? (yes/no or y/n): ").strip().lower()
        
        if should_download in ['yes', 'y']:
            # Create the directory structure if it doesn't exist
            os.makedirs(os.path.dirname(file["destination"]), exist_ok=True)

            # Download the file using the defined function
            try:
                download_file_from_google_drive(file["file_id"], file["destination"])
                logging.info(f"Download completed. The file has been saved to: {file['destination']}")
            except Exception as e:
                logging.error(f"An error occurred while downloading the file: {e}")
            break
        elif should_download in ['no', 'n']:
            logging.info(f"Skipped downloading the file to: {file['destination']}")
            break
        else:
            logging.warning("Invalid input. Please enter 'yes', 'no', 'y', or 'n'.")
