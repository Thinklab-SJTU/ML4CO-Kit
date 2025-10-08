r"""
The utilities used to download and compress files.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import os
import uuid
import time
import shutil
import requests
import aiohttp
import asyncio
import hashlib
import zipfile
import tarfile
import async_timeout
import urllib.request
from tqdm import tqdm
from huggingface_hub import HfApi


###############################################
#                  Download                   #
###############################################

def download(file_path: str, url: str, md5: str = None, retries: int = 5):
    r"""
    The specific download function with three different methods.
    """
    if retries <= 0:
        raise RuntimeError("Max Retries exceeded!")
    if not os.path.exists(file_path):
        check_file_path(file_path)
        print(f"\nDownloading to {file_path}...")
        if retries % 3 == 1:
            try:
                down_res = requests.get(url, stream=True)
                file_size = int(down_res.headers.get("Content-Length", 0))
                with tqdm.wrapattr(down_res.raw, "read", total=file_size) as content:
                    with open(file_path, "wb") as file:
                        shutil.copyfileobj(content, file)
            except requests.exceptions.ConnectionError as err:
                print("Warning: Network error. Retrying...\n", err)
                return download(file_path, url, md5, retries - 1)
        elif retries % 3 == 2:
            try:
                asyncio.run(_asyncdownload(file_path, url))
            except:
                return download(file_path, url, md5, retries - 1)
        else:
            try:
                urllib.request.urlretrieve(url, file_path)
            except:
                return download(file_path, url, md5, retries - 1)

    if md5 is not None:
        md5_returned = get_md5(file_path)
        if md5 != md5_returned:
            print("Warning: MD5 check failed for the downloaded content. Retrying...")
            os.remove(file_path)
            time.sleep(1)
            return download(file_path, url, md5, retries - 1)
    return file_path


async def _asyncdownload(file_path: str, url: str):
    r"""
    The asynchronous method to download data.
    """
    async with aiohttp.ClientSession() as session:
        async with async_timeout.timeout(120):
            async with session.get(url) as response:
                with open(file_path, "wb") as file:
                    async for data in response.content.iter_chunked(512):
                        file.write(data)



def get_md5(file_path: str):
    r"""
    Get the md5 of the specified file.
    """
    hash_md5 = hashlib.md5()
    chunk = 8192
    with open(file_path, "rb") as file_to_check:
        while True:
            buffer = file_to_check.read(chunk)
            if not buffer:
                break
            hash_md5.update(buffer)
        md5_returned = hash_md5.hexdigest()
        return md5_returned


def pull_file_from_huggingface(
    repo_id: str, repo_type: str, filename: str, save_path: str, hf_token: str = None
):
    # cache and local
    name = uuid.uuid4().hex
    root_dir = f"tmp/{name}"
    local_dir = f"tmp/{name}/local"
    cache_dir = f"tmp/{name}/cache"
    download_path = os.path.join(local_dir, filename)
    
    # download
    hf_api = HfApi(token=hf_token)
    hf_api.hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    
    # check save path
    save_path_dir = os.path.dirname(save_path)
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)

    # move file
    shutil.move(download_path, save_path)
    
    # delete root dir
    shutil.rmtree(root_dir)


###############################################
#            Compress and Extract             #
###############################################


def compress_folder(
    folder: str, compress_path: str,
):
    """
    Compresses a folder into the specified output format.
    Raise ``ValueError`` if the out put file format is un supported.
    
    :param folder: string, the path to the folder to be compressed.
    :param compress_path: string, the path to the output file.

    Supported formats:
        - .zip: ZIP format
        - .tar.gz: tar.gz format

    .. dropdown:: Example
        
        ::
        
            # We also use tsp_uniform for illustration
            # if you haven't download it, please download it as mentioned above.
            >>>from ml4co_kit import compress_folder
            
            # Compress the folder
            >>>compress_folder("dataset/tsp_uniform_20240825","dataset/tsp_uniform_20240825.tar.gz")
    """
    if compress_path.endswith(".zip"):
        shutil.make_archive(compress_path[:-4], "zip", folder)
    elif compress_path.endswith(".tar.gz"):
        with tarfile.open(compress_path, "w:gz") as tar:
            tar.add(folder, arcname="")
    else:
        message = "Unsupported file format. Only .zip, .tar.gz"
        raise ValueError(message)


def extract_archive(archive_path: str, extract_path: str):
    """
    Extracts an archive into the specified extract path.
    Raise ``ValueError`` if the out put file format is un supported.

    :param archive_path: string, path to the archive file.
    :param extract_path: string, path to the extraction directory.

    Supported formats:
        - .zip: ZIP format
        - .tar.gz: tar.gz format

    .. dropdown:: Example
        
        ::
        
            # We also use tsp_uniform for illustration
            # if you haven't download it, please download it as mentioned above.
            >>> from ml4co_kit import extract_archive
            
            # Extracts the archive
            >>> extract_archive("dataset/tsp_uniform_20240825.tar.gz","dataset/tsp_uniform_20240825")
    """
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
    elif archive_path.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_path)
    else:
        message = "Unsupported file format. Only .zip, .tar.gz"
        raise ValueError(message)
    
    
###############################################
#                  Check Path                 #
###############################################

def check_file_path(file_path: str):
    r"""
    Check if the directory of the file path exists, if not, create it.
    """
    # Get Directory
    directory = os.path.dirname(file_path)
    
    # Create Directory if it does not exist
    if not directory == "":
        os.makedirs(directory, exist_ok=True)