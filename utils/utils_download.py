from zipfile import ZipFile
import os

import gdown


def download_datasets():
    # a file
    url = "https://drive.google.com/file/d/1odl_ChAyzPsmZ5ZQ-VOBwOeZVsBB4EL5/view?usp=sharing"

    output = "datasets/datsets.zip"
    gdown.download(url, output, quiet=False, fuzzy=True)

    return output


# unzip files in folder
def unzip_files(destination):

    # unzip files
    with ZipFile(destination, "r") as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(path=os.path.dirname(destination))

    # remove zip file
    # os.remove(destination)
