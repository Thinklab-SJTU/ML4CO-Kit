import os
import sys
import wget
import shutil


# pyconcorde
ori_dir = os.getcwd()
os.chdir('data4co/solver/tsp/pyconcorde')
os.system("python ./setup.py build_ext --inplace")
os.chdir(ori_dir)

# LKH
lkh_url = "http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.7.tgz"
wget.download(url=lkh_url, out="LKH-3.0.7.tgz")
os.system("tar xvfz LKH-3.0.7.tgz")
ori_dir = os.getcwd()
os.chdir('LKH-3.0.7')
os.system("make")
target_dir = os.path.join(sys.prefix, "bin")
os.system(f"cp LKH {target_dir}")
os.chdir(ori_dir)
os.remove("LKH-3.0.7.tgz")
shutil.rmtree("LKH-3.0.7")

# KaMIS
shutil.copytree('data4co/solver/mis/kamis-source/', 
                'data4co/solver/mis/KaMIS/tmp_build/')
ori_dir = os.getcwd()
os.chdir('data4co/solver/mis/KaMIS/tmp_build')
os.system("bash cleanup.sh")
os.system("bash compile_withcmake.sh")
os.chdir(ori_dir)
shutil.copytree('data4co/solver/mis/KaMIS/tmp_build/deploy/', 
                'data4co/solver/mis/KaMIS/deploy/')
shutil.rmtree('data4co/solver/mis/KaMIS/tmp_build/')