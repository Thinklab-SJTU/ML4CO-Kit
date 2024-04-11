import os
import shutil


# pyconcorde
ori_dir = os.getcwd()
os.chdir("ml4co_kit/solver/tsp/pyconcorde")
os.system("python ./setup.py build_ext --inplace")
os.chdir(ori_dir)

# KaMIS
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
shutil.rmtree("ml4co_kit/solver/mis/KaMIS/tmp_build/")
