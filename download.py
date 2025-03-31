import kagglehub
path = kagglehub.dataset_download("wildlifedatasets/wildlifereid-10k")
path2 = kagglehub.dataset_download("wildlifedatasets/wildlifereid-10k-features")
print("Path to dataset files:", path)
print("Path to feature files:", path2)