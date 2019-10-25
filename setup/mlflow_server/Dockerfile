FROM continuumio/miniconda3

ARG MLFLOW_VERSION

RUN echo "\n###\n# BUILDING mlflow_server image with mlflow ${MLFLOW_VERSION}\n###"

# Add Postgres SQL specific packages
RUN pip install mlflow==${MLFLOW_VERSION} sqlalchemy==1.3.0 psycopg2-binary

RUN pip install minio boto3

COPY wait-for-pgdb-and-minio.sh /
COPY test_pgdb_connection.py /
COPY initialize_bucket.py /