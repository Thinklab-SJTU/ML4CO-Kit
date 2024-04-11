import os
import sys
import shutil

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from ml4co_kit.utils.file_utils import compress_folder, extract_archive


def test_file_utils():
    # compress .tar.gz
    compress_folder(
        folder="tests/utils_test/file_utils_test",
        compress_path="tests/utils_test/file_utils_test.tar.gz"
    )
    shutil.rmtree("tests/utils_test/file_utils_test")
    
    # extract_archive .tar.gz
    extract_archive(
        archive_path="tests/utils_test/file_utils_test.tar.gz",
        extract_path="tests/utils_test/file_utils_test"
    )
    os.remove("tests/utils_test/file_utils_test.tar.gz")
    
    # compress .zip
    compress_folder(
        folder="tests/utils_test/file_utils_test",
        compress_path="tests/utils_test/file_utils_test.zip"
    )
    shutil.rmtree("tests/utils_test/file_utils_test")
    
    # extract_archive .zip
    extract_archive(
        archive_path="tests/utils_test/file_utils_test.zip",
        extract_path="tests/utils_test/file_utils_test"
    )
    os.remove("tests/utils_test/file_utils_test.zip")


if __name__ == "__main__":
    test_file_utils()
