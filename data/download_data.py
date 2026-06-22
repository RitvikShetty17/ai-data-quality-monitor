import urllib.request
import os

# NYC Yellow Taxi January 2024 - official NYC TLC dataset
# Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
OUTPUT_PATH = "data/yellow_tripdata_2024-01.parquet"

def download():
    if os.path.exists(OUTPUT_PATH):
        print(f"File already exists at {OUTPUT_PATH}, skipping download.")
        return

    print("Downloading NYC Yellow Taxi dataset (Jan 2024)...")
    print("This is ~50MB, may take a minute...")
    urllib.request.urlretrieve(URL, OUTPUT_PATH)
    print(f"Done! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    download()