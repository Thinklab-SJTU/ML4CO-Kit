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
from typing import Union


###############################################
#                  Download                   #
###############################################


def download(filename: str, url: Union[str, list], md5: str = None, retries: int = 5):
    r"""
    Download data form url and check md5.
    Raise ``RuntimeError`` if exceeded the maximum number of attempts.
    Raise ``ValueError`` if ``url``is neither string nor list.
 
    :param filename: string, the name of the file need to download.
    :param url: string or list, the url path of the data set.
    :param md5: string, the md5 used to check the md5 of download data.
    :param retries: int, the max accepectable retry count. Defaults to '5'.

    ..dropdown:: Example
    
        ::
            # we try to download tsp_uniform dataset as example.
            >>> from ml4co_kit import download
            
            #create downloader and load data from huggingface.co
            >>> download("tsp_uniform_20240825.tar.gz",""https://huggingface.co/datasets/ML4CO/TSPUniformDataset/resolve/main/tsp_uniform_20240825.tar.gz?download=true,"44371d7c99b35d77fe18220122c564c1")
            
            #The data will be stored in the specified path if the download is successful.
    """
    if type(url) == str:
        return _download(filename, url, md5, retries)
    elif type(url) == list:
        for cur_url in url:
            try:
                return _download(filename, cur_url, md5, retries)
            except RuntimeError:
                continue
        raise RuntimeError("Max Retries exceeded!")
    else:
        raise ValueError("The url should be string or list of string.")


async def _asyncdownload(filename: str, url: str):
    r"""
    The asynchronous method to download data.
    """
    async with aiohttp.ClientSession() as session:
        async with async_timeout.timeout(120):
            async with session.get(url) as response:
                with open(filename, "wb") as file:
                    async for data in response.content.iter_chunked(512):
                        file.write(data)


def _download(filename: str, url: str, md5: str, retries: int):
    r"""
    The specific download function with three different methods.
    """
    if retries <= 0:
        raise RuntimeError("Max Retries exceeded!")
    if not os.path.exists(filename):
        print(f"\nDownloading to {filename}...")
        if retries % 3 == 1:
            try:
                down_res = requests.get(url, stream=True)
                file_size = int(down_res.headers.get("Content-Length", 0))
                with tqdm.wrapattr(down_res.raw, "read", total=file_size) as content:
                    with open(filename, "wb") as file:
                        shutil.copyfileobj(content, file)
            except requests.exceptions.ConnectionError as err:
                print("Warning: Network error. Retrying...\n", err)
                return download(filename, url, md5, retries - 1)
        elif retries % 3 == 2:
            try:
                asyncio.run(_asyncdownload(filename, url))
            except:
                return _download(filename, url, md5, retries - 1)
        else:
            try:
                urllib.request.urlretrieve(url, filename)
            except:
                return _download(filename, url, md5, retries - 1)

    if md5 is not None:
        md5_returned = _get_md5(filename)
        if md5 != md5_returned:
            print("Warning: MD5 check failed for the downloaded content. Retrying...")
            os.remove(filename)
            time.sleep(1)
            return _download(filename, url, md5, retries - 1)
    return filename


def _get_md5(filename: str):
    r"""
    Get the md5 of the specified file.
    """
    hash_md5 = hashlib.md5()
    chunk = 8192
    with open(filename, "rb") as file_to_check:
        while True:
            buffer = file_to_check.read(chunk)
            if not buffer:
                break
            hash_md5.update(buffer)
        md5_returned = hash_md5.hexdigest()
        return md5_returned


###############################################
#                  Compress                   #
###############################################


def compress_folder(
    folder: str,
    compress_path: str,
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
        
            #we also use tsp_uniform for illustration
            #if you haven't download it, please download it as mentioned above.
            >>>from ml4co_kit import compress_folder
            
            #compress the folder
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
        
            #we also use tsp_uniform for illustration
            #if you haven't download it, please download it as mentioned above.
            >>> from ml4co_kit import extract_archive
            
            #Extracts the archive
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
