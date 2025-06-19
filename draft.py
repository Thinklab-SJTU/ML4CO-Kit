from ml4co_kit import OPSolver

solver = OPSolver()
solver.from_txt(file_path="/home/zhanghang/chennuoyan/data/op/op_const20_test_seed1234.txt")
solver.to_txt(file_path="/home/zhanghang/chennuoyan/data/op/op_const20_test_seed1234_output.txt")