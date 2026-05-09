import os
os.environ['KAGGLE_CONFIG_DIR'] = r'C:\Users\abbas\.kaggle'

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
print("Authentication successful!")

# Download only the instances_val2017.json annotation file
print("Downloading annotations...")
api.dataset_download_file(
    'awsaf49/coco-2017-dataset',
    file_name='coco2017/annotations/instances_val2017.json',
    path='data/annotations'
)
print("Annotations downloaded!")