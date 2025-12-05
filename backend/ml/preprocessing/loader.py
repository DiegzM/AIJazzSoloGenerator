import pandas as pd
from rich.console import Console
from config.settings import RAW_DATA_DIR

console = Console()

# Required files for data loading

REQUIRED_FILES = {
    "melody": "melody.csv",
    "beats": "beats.csv",
    "sections": "sections.csv",
    "solo_info": "solo_info.csv",
}

# Check if all required files exist
def _check_if_files_exist():
    missing_files = []
    for file_key, file_name in REQUIRED_FILES.items():
        file_path = RAW_DATA_DIR / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    if missing_files:
        console.print("\n[red]Missing required files:[/red]\n")
        for file in missing_files:
            console.print(f" - {file}")
        raise FileNotFoundError()

# Load datasets into a dictionary
def _load_datasets():
    _check_if_files_exist()
    
    datasets = {}
    for file_key, file_name in REQUIRED_FILES.items():
        file_path = RAW_DATA_DIR / file_name
        datasets[file_key] = pd.read_csv(file_path)

    console.print("\n[green]All required files found. Datasets loaded successfully.[/green]")
    
    return datasets

def load():
    datasets = _load_datasets()
    return datasets