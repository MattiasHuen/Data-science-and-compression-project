import kagglehub

download_dir = "src/data"   # or "./data/div2k" on Windows: r"C:\data\div2k"

path = kagglehub.dataset_download(
    "takihasan/div2k-dataset-for-super-resolution",
)

print("Path to dataset files:", path)