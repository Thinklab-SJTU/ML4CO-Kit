import os
import shutil
import pathlib

kamis_path = pathlib.Path(__file__).parent

if os.path.exists(kamis_path / "KaMIS/deploy/"):
    shutil.rmtree(kamis_path / "KaMIS/deploy/")
if os.path.exists(kamis_path / "KaMIS/tmp_build/"):
    shutil.rmtree(kamis_path / "KaMIS/tmp_build/")
shutil.copytree(
    kamis_path / "kamis-source/", kamis_path / "KaMIS/tmp_build/"
)
ori_dir = os.getcwd()
os.chdir(kamis_path / "KaMIS/tmp_build/")
os.system("bash cleanup.sh")
os.system("bash compile_withcmake.sh")
os.chdir(ori_dir)
shutil.copytree(
    kamis_path / "KaMIS/tmp_build/deploy/",
    kamis_path / "KaMIS/deploy/",
)
shutil.rmtree(kamis_path / "KaMIS/tmp_build/")