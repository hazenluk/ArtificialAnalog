from pathlib import Path

def get_project_path():
    path = Path.cwd()
    for _ in range(5):
        if path.name == "Artificial Analog":
            return path
        path = path.parent
    print("Error: Couldn't Find Project Root Directory \"Artificial Analog\" in path tree above CWD")