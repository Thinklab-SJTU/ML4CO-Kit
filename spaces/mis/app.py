import os
import time
import shutil
import gradio as gr
from ml4co_kit import KaMISSolver, draw_mis_problem, draw_mis_solution


MIS_DEFAULT_PATH = "assets/mis_default.png"
MIS_PROBLEM_PATH = "assets/mis_problem.png"
MIS_SOLUTION_PATH = "assets/mis_solution.png"
GPICKLE_PATH = "tmp/mis_problem.gpickle"
RESULT_PATH = "tmp/solve/mis_problem_unweighted.result"


def _handle_mis_solve(file_path: str):
    # Create a blank temporary folder
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    else:
        shutil.rmtree("tmp")
        os.mkdir("tmp")
        
    # Move the file to the temporary folder
    shutil.move(file_path, GPICKLE_PATH)
    
    # Begin solve and record the solving time
    start_time = time.time()
    solver = KaMISSolver()
    solver.solve("tmp")
    solved_time = time.time() - start_time
    
    # Draw pictures
    draw_mis_problem(
        save_path=MIS_PROBLEM_PATH,
        gpickle_path=GPICKLE_PATH
    )
    draw_mis_solution(
        save_path=MIS_SOLUTION_PATH,
        gpickle_path=GPICKLE_PATH,
        result_path=RESULT_PATH,
        pos_type="kamada_kawai_layout"
    )
    
    # Message
    message = "Successfully solve the MIS problem, using time ({:.3f}s).".format(solved_time)
    
    return message, MIS_PROBLEM_PATH, MIS_SOLUTION_PATH
    

def handle_mis_solve(file_path: str):
    try:
        message = _handle_mis_solve(file_path)
        return message
    except Exception as e:
        message = str(e)
        return message, MIS_PROBLEM_PATH, MIS_SOLUTION_PATH


def handle_mis_clear():
    # Replace the original image with the default image
    shutil.copy(
        src=MIS_DEFAULT_PATH,
        dst=MIS_PROBLEM_PATH
    )
    shutil.copy(
        src=MIS_DEFAULT_PATH,
        dst=MIS_SOLUTION_PATH
    )
    message = "successfully clear the files!"
    return message, MIS_PROBLEM_PATH, MIS_SOLUTION_PATH


def convert_image_path_to_bytes(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return image_bytes


with gr.Blocks() as mis_page:

    gr.Markdown(
        '''
        This space displays the solution to the MIS problem.

        ## How to use this Space?
        - Upload a '.gpickle' file.
        - The images of the MIS problem and solution will be shown after you click the solve button.
        - Click the 'clear' button to clear all the files.
        ## Examples
        - You can get the test examples from our [MIS Dataset Repo.](https://huggingface.co/datasets/SJTU-TES/MIS)
        '''
    )

    with gr.Row(variant="panel"):
        with gr.Column(scale=7):
            with gr.Row():
                mis_file = gr.File(
                    file_types=[".gpickle"],
                    scale=3
                )
                info = gr.Textbox(
                    value="",
                    label="Log",
                    scale=4,
                )
        with gr.Column(scale=4):
            mis_problem_img = gr.Image(
                value="assets/mis_problem.png", 
                type="filepath", 
                label="MIS Problem", 
            )
        with gr.Column(scale=4):
            mis_solution_img = gr.Image(
                value="assets/mis_solution.png", 
                type="filepath", 
                label="MIS Solution", 
            )
    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            solve_button = gr.Button(
                value="Solve", 
                variant="primary", 
                scale=1
            )
        with gr.Column(scale=1, min_width=100):
            clear_button = gr.Button(
                "Clear", 
                variant="secondary", 
                scale=1
            )
        with gr.Column(scale=8):
            pass         
    
    solve_button.click(
        handle_mis_solve,
        [mis_file],
        outputs=[info, mis_problem_img, mis_solution_img]
    )
    
    clear_button.click(
        handle_mis_clear,
        inputs=None,
        outputs=[info, mis_problem_img, mis_solution_img]
    )
    

if __name__ == "__main__":
    mis_page.launch(debug = True)