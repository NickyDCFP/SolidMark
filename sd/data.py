import pandas as pd
import random
from time import time
import json
LAION_SUBSET_SIZE = 200000 / 0.7 # usually 30% of URLs are broken
NUM_PARQUETS = 128

random.seed(time())
print("Loading in LAION sample lists.")
start = time()
subsets = []
for i in range(NUM_PARQUETS):
    print(f"Loading in list ID {i}")
    subsets.append(pd.read_parquet(f"dataset/laion/laion-2B/relaion2B-en-research-safe/part-{str(i).zfill(5)}-339dc23d-0869-4dc0-9390-b4036fcc80c2-c000.snappy.parquet", engine="pyarrow").sample(LAION_SUBSET_SIZE // NUM_PARQUETS + 1, axis=0, ignore_index=True))
    print(f"Elapsed time: {time() - start} seconds")
data_df = pd.concat(subsets, axis=0, ignore_index=True)
print("Total rows:", len(data_df))
print("DataFrame Head:", data_df.head())

data_df.to_parquet('dataset/laion/laion-2B/subset_200k_descriptor.parquet', engine='pyarrow')

