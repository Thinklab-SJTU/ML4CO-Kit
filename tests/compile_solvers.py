import os
import sys
import shutil
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from ml4co_kit import download


# LKH
lkh_url = "http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.7.tgz"
download("LKH-3.0.7.tgz", url=lkh_url)
os.system("tar xvfz LKH-3.0.7.tgz")
ori_dir = os.getcwd()
os.chdir("LKH-3.0.7")
os.system("make")
target_dir = os.path.join(sys.prefix, "bin")
os.system(f"cp LKH {target_dir}")
os.chdir(ori_dir)
os.remove("LKH-3.0.7.tgz")
shutil.rmtree("LKH-3.0.7")


# KaMIS
if os.path.exists("ml4co_kit/solver/mis/KaMIS"):
    shutil.rmtree("ml4co_kit/solver/mis/KaMIS")
shutil.copytree(
    "ml4co_kit/solver/mis/kamis-source/", "ml4co_kit/solver/mis/KaMIS/tmp_build/"
)
ori_dir = os.getcwd()
os.chdir("ml4co_kit/solver/mis/KaMIS/tmp_build")
os.system("bash cleanup.sh")
os.system("bash compile_withcmake.sh")
os.chdir(ori_dir)
shutil.copytree(
    "ml4co_kit/solver/mis/KaMIS/tmp_build/deploy/", "ml4co_kit/solver/mis/KaMIS/deploy/"
)
shutil.rmtree("ml4co_kit/solver/mis/KaMIS/tmp_build")
