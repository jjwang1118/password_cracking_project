from pathlib import Path

def config():
    dataset_name = "CompilationOfManyBreaches"
    return {
        "dataset_name": dataset_name,
        "leak_dataset_path": str(
            Path.home()
            / "projects"
            / "leak_datasets"
            / "CompilationOfManyBreaches"
            / "data"
        ),
        "max_len": 20,
        "min_len": 8,
        "password_map" : "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./;<=>?@[\\]^_`{|}~ ",
        "output_path": str(
            Path.home()
            / "projects"
            / "password_cracking_project"
            / "datasets_filtered"
            / f"{dataset_name}_filtered"
        ),
        "stastic_path" : str(
            Path.home()
            / "projects"
            / "password_cracking_project"
            / "datasets_filtered"
            / "stastic.csv"
        )
    }
