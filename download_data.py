import requests
import os

def download_file_from_google_drive(file_id, dest_path):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    # Handle large file confirmation
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print(f"Download complete: {dest_path}")

if __name__ == "__main__":
    FILE_ID = "1q8Koz4y0Cd4NCHli7zOuUS58OEokMriT"
    DESTINATION = "data/Reviews.csv"
    download_file_from_google_drive(FILE_ID, DESTINATION)
