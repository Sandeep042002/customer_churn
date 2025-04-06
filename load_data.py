import pandas as pd
import os
import shutil
import logging

# from utils import get_logger

# logger = get_logger(__name__)

# Read the CSV file
df = pd.read_csv("customer_call.csv")

# Create the data directory if it doesn't exist
data_path = "Data"
os.makedirs(data_path, exist_ok=True)

file_path = os.path.join(data_path, "customer_call.csv")


shutil.copy("customer_call.csv", file_path)

# logger.info(f"File saved to: {file_path}")
