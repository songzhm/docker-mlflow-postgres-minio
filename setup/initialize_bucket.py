# -*- coding: utf-8 -*-
"""
Created by: zhengmingsong
Created on: 10/23/19 2:19 AM
"""

from minio import Minio
import os
import sys

try:
    print(os.environ['MLFLOW_S3_ENDPOINT_URL'])
    minioClient = Minio('minio:{}'.format(os.environ['MINIO_PORT']),
                        access_key=os.environ['MINIO_ACCESS_KEY'],
                        secret_key=os.environ['MINIO_SECRET_KEY'],
                        secure=False)
    if minioClient.bucket_exists(os.environ['MLFLOW_BUCKET_NAME']):
        pass
    else:
        minioClient.make_bucket(os.environ['MLFLOW_BUCKET_NAME'])
except Exception as e:
    print(str(e))
    sys.exit(1)