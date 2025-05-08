import os
from nbformat import read, write

def clear_notebook_outputs(notebook_path):
    # change
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = read(f, as_version=4)

    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None

    with open(notebook_path, 'w', encoding='utf-8') as f:
        write(os.path.join("notebooks","test.ipynb"), f)

# Replace 'your_notebook.ipynb' with the path to your notebook
clear_notebook_outputs("D:\\code\\StreamTSAD\\notebooks\\Datasets_Analysis.ipynb")
