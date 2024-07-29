import requests

# URL of the file to download
url = "https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf"

# Name of the file to save
file_name = "openhermes-2.5-mistral-7b.Q4_K_M.gguf"

# Download the file
response = requests.get(url)
if response.status_code == 200:
    with open(file_name, 'wb') as f:
        f.write(response.content)
    print(f"File '{file_name}' downloaded successfully.")
else:
    print(f"Error downloading the file: {response.status_code}")