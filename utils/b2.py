import pandas as pd
import boto3
from botocore.config import Config
from io import BytesIO
import chardet


def detect_encoding(content):
    result = chardet.detect(content)
    return result['encoding']


class B2:
    def __init__(self, endpoint, key_id, secret_key):
        """
        Set up a connection to the Backblaze S3 bucket.

        Parameters
        ----------
        endpoint : str
            The endpoint URL, typically starting with "https://s3. ...".
        key_id : str
            The application key ID for Backblaze.
        secret_key : str
            The application key secret for Backblaze.
        """
        self.b2 = boto3.resource(
            service_name='s3',
            endpoint_url=endpoint,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4')
        )

    def set_bucket(self, bucket_name):
        """
        Select a bucket for operations.

        Parameters
        ----------
        bucket_name : str
            The name of the bucket.
        """
        self.bucket = self.b2.Bucket(bucket_name)

    def list_files(self):
        """
        List all files in the bucket.

        Returns
        -------
        list of str
            A list of file keys.
        """
        return [f.key for f in self.bucket.objects.all()]

    def get_df(self, remote_path):
        """
        Retrieve a CSV file from the bucket and return it as a DataFrame.
        Automatically detects the file's encoding.

        Parameters
        ----------
        remote_path : str
            The path to the file in the bucket.

        Returns
        -------
        pd.DataFrame
            The content of the CSV file as a DataFrame.
        """
        obj = self.bucket.Object(remote_path)
        content = obj.get()['Body'].read()

        # Detect encoding
        encoding = detect_encoding(content)

        # Load CSV with the detected encoding
        return pd.read_csv(BytesIO(content), encoding=encoding)

