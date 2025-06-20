from ml4co_kit import OPGurobiSolver

solver = OPGurobiSolver()
# solver.from_txt(file_path="/home/zhanghang/chennuoyan/data/op/op_const20_test_seed1234.txt")
# obj_values, tours = solver.solve()
# print(obj_values)
# print(tours)
# solver.to_txt(file_path="/home/zhanghang/chennuoyan/data/op/op_const20_test_seed1234_output.txt")

solver.from_txt(file_path="/home/zhanghang/chennuoyan/data/op/op_const20_test_seed1234.txt")
total_costs, tours = solver.solve(show_time=True)
print(total_costs)
print(tours)
solver.to_txt(file_path="/home/zhanghang/chennuoyan/data/op/op_const20_test_seed1234_output.txt")