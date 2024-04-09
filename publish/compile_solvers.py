import os
import shutil


# pyconcorde
ori_dir = os.getcwd()
os.chdir("data4co/solver/tsp/pyconcorde")
os.system("python ./setup.py build_ext --inplace")
os.chdir(ori_dir)

# KaMIS
shutil.copytree(
    "data4co/solver/mis/kamis-source/", "data4co/solver/mis/KaMIS/tmp_build/"
)
ori_dir = os.getcwd()
os.chdir("data4co/solver/mis/KaMIS/tmp_build")
os.system("bash cleanup.sh")
os.system("bash compile_withcmake.sh")
os.chdir(ori_dir)
shutil.copytree(
    "data4co/solver/mis/KaMIS/tmp_build/deploy/", "data4co/solver/mis/KaMIS/deploy/"
)
shutil.rmtree("data4co/solver/mis/KaMIS/tmp_build/")
