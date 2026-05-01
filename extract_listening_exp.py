from pathlib import Path
from shutil import copy2


if __name__ == "__main__":
    cur_dir = Path(__file__).parent.resolve()
    results_dir = cur_dir / "raw_data"
    uc_results_path = results_dir / "UChicago"
    ob_results_path = results_dir / "Oberlin"
    uc_output = cur_dir / "listening_experience" / "UChicago"
    ob_output = cur_dir / "listening_experience" / "Oberlin"
    uc_output.mkdir(parents = True, exist_ok = True)
    ob_output.mkdir(parents = True, exist_ok = True)


    rows = []
    for i, experimental_site_path in enumerate([uc_results_path, ob_results_path]):
        site = "UChicago" if i == 0 else "Oberlin"
        if site == "UChicago":
            dest_folder = uc_output
        else:
            dest_folder = ob_output
        for subject_folder_path in experimental_site_path.iterdir():
            subject_id = subject_folder_path.name
            fname = f"listening_experience_{subject_id}.txt"
            list_exp_path = subject_folder_path / fname 
            try:
                copy2(list_exp_path, dest_folder / fname)
            except:
                print(f"No listening experience file found!\n  Subject Id: {subject_id}\n  Site: {site}")

            