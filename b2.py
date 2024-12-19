import mimetypes 

import pandas as pd
import boto3
from botocore.config import Config
import chardet
from io import BytesIO


class B2(object):
    def __init__(self, endpoint, key_id, secret_key):
        """
        Set up a connection between the current instance and Backblaze.

        Parameters
        ----------
        endpoint : str
            The endpoint, usually starting with "https://s3. ..."
        key_id : str
            The "Key ID" for the application key from Backblaze.
        secret_key : str
            The Key secret, or "Key" for the Backblaze app key itself.
        """
        # Return a boto3 resource object for B2 service
        self.b2 = boto3.resource(service_name='s3',
                                endpoint_url=endpoint,
                                aws_access_key_id=key_id,
                                aws_secret_access_key=secret_key,
                                config=Config(signature_version='s3v4'))
        
    def set_bucket(self, bucket_name):
        """
        Select a bucket accessible by the chosen app key.

        Parameters
        ----------
        bucket_name : str
            Name of Bucket
        """
        self.bucket = self.b2.Bucket(bucket_name)

    def list_files(self, verbose=False):
        if verbose:
            return [f.get() for f in self.bucket.objects.all()]
        else:
            return [f.key for f in self.bucket.objects.all()]

    def get_df(self, remote_path):
        # Get file
        obj = self.bucket.Object(remote_path)
        # Get the raw content of the file
        content = obj.get()['Body'].read()
        # Load the content as a DataFrame from the pickle file
        df = pd.read_pickle(BytesIO(content))
        return df
    
    def get_model(self, remote_path):
       # Get file
        obj = self.bucket.Object(remote_path)
        # Get the raw content of the file
        df = obj.get()['Body'].read()
        return df
            
    def get_object(self, remote_path):
        obj = self.bucket.Object(remote_path)
        return obj.get()['Body']

    def file_to_b2(self, local_path, remote_path):
        '''
        Send `local_path` file to `remote_path`.
        '''
        # Guess the type of a file based on its URL
        mimetype, _ = mimetypes.guess_type(local_path)

        if mimetype is None:
            raise Exception("Failed to guess mimetype")
        
        if remote_path in [f.key for f in self.bucket.objects.all()]:
            print(f'Overwriting {remote_path} ...')
        else:
            print(f'Uploading {remote_path} ...')
        
        self.bucket.upload_file(
            Filename=local_path,
            Key=remote_path,
            ExtraArgs={
                "ContentType": mimetype
            }
        )
    
