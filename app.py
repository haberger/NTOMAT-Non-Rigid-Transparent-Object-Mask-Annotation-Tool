import gradio as gr
import argparse

def manage_files(files):
    names = [file.name for file in files]
    names_str = ", ".join(names)
    return "Files received: " + names_str



def main(dataset_path):

    
    with gr.Blocks() as demo:
        gr.Markdown("## Select a Folder")
        folder_input = gr.File(label="Choose Folder", file_count="directory")
        output = gr.Textbox()
        folder_input.change(manage_files, folder_input, output)
    demo.launch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NTOMAT')
    parser.add_argument(  
        '-d',
        dest='dataset_path',
        help='path_to_3D-DAT Dataset')

    args = parser.parse_args()

    main(args.dataset_path)