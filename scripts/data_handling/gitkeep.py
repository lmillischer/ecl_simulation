import os


def add_gitkeep_to_empty_folders(directory):
    for root, dirs, files in os.walk(directory):
        # Check if the directory is empty
        if os.listdir(root):
            gitkeep_path = os.path.join(root, '.gitkeep')
            print(f'Adding {gitkeep_path}')
            open(gitkeep_path, 'a').close()

# Replace 'your_project_directory' with the path to your project
add_gitkeep_to_empty_folders('../../plots')
