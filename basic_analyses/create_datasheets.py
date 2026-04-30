from pathlib import Path
import csv

def score_subject_questionnaires(subject_folder_path):

    # Grab subject number
    subject_number = subject_folder_path.name

    # Order of presentation is..
    #   1. flow_state_scale_{subject_number}.csv
    #   2. tellegen_{subject_number}.csv
    #   3. vhq_{subject_number}.csv
    #   4. launay_slade_{subject_number}.csv
    #   5. dissociative_experiences_{subject_number}.csv
    #   6. bais_v_{subject_number}.csv
    #   7. bais_c_{subject_number}.csv
    questionnaire_names = [
        "flow_state_scale",
        "tellegen",
        "vhq",
        "launay_slade",
        "dissociative_experiences",
        "bais_v",
        "bais_c",
    ]

    subject_questionnaire_scores = []
    for questionnaire_name in questionnaire_names:
        file_path = subject_folder_path / f"{questionnaire_name}_{subject_number}.csv"

        with open(file_path, mode = "r", encoding = "utf8") as f:
            reader = csv.reader(f)
            lines = list(reader)

        answers = lines[1][1:]
        raw_sum = sum(int(value) for value in answers)
        subject_questionnaire_scores.append(raw_sum)

    flow_state_scale_raw_sum = subject_questionnaire_scores[0]
    tellegen_raw_sum = subject_questionnaire_scores[1]
    vhq_raw_sum = subject_questionnaire_scores[2]
    launay_slade_raw_sum = subject_questionnaire_scores[3]
    dissociative_experiences_raw_sum = subject_questionnaire_scores[4]
    bais_v_raw_sum = subject_questionnaire_scores[5]
    bais_c_raw_sum = subject_questionnaire_scores[6]

    derived_questionnaire_scores = [
        round(flow_state_scale_raw_sum / 9, 4),
        round(tellegen_raw_sum / 34, 4),
        round(vhq_raw_sum / 14, 4),
        round(launay_slade_raw_sum / 16, 4),
        round(dissociative_experiences_raw_sum / 28, 4),
        round(bais_v_raw_sum / 14, 4),
        round(bais_c_raw_sum / 14, 4),
    ]

    return subject_questionnaire_scores + derived_questionnaire_scores
    
def run_one_subject(subject_folder_path):

    # Grab subject number
    subject_number = subject_folder_path.name

    with open(subject_folder_path / f"stanford_sleepiness_{subject_number}.csv", mode = "r", encoding = "utf8") as f:
        reader = csv.reader(f)
        lines = list(reader)

    sleepiness_scores = {}
    for line in lines[1:]:
        block_scheme = line[2]
        pre_or_post = line[3]
        response = int(line[5])
        sleepiness_scores[(block_scheme, pre_or_post)] = response
    

    # Store which block came first
    with open(subject_folder_path / f"summary_data_{subject_number}.csv", mode = "r", encoding = "utf8") as f:
        reader = csv.reader(f)
        lines = list(reader)
        block1_name = lines[1][1]
    
    if block1_name == "full_sentence":
        full_sentence_order = 1
        imagined_sentence_order = 2
    else:
        full_sentence_order = 2
        imagined_sentence_order = 1
    
    # Get hits, misses, false alarms, false postives for each block
    for trial_type in ["full_sentence", "imagined_sentence"]:
        hits = 0
        false_alarms = 0
        total_targets = 0
        total_distractors = 0
        with open(subject_folder_path / trial_type / f"{trial_type}_{subject_number}.csv") as f:
            reader = csv.reader(f)
            lines = list(reader)
            data_lines = lines[1:]

            for line in data_lines:
                stim_type = line[3]
                response = line[4]

                if stim_type == "target":
                    total_targets += 1
                    if response == "target":
                        hits += 1
                else:
                    total_distractors += 1
                    if response == "target":
                        false_alarms += 1
        if trial_type == "full_sentence":
            full_sentence_data = [
                full_sentence_order,
                total_targets,
                total_distractors,
                hits,
                false_alarms,
                sleepiness_scores[("full_sentence", "pre")],
                sleepiness_scores[("full_sentence", "post")],
                sleepiness_scores[("full_sentence", "post")] - sleepiness_scores[("full_sentence", "pre")],
            ]
        else:
            imagined_sentence_data = [
                imagined_sentence_order,
                total_targets,
                total_distractors,
                hits,
                false_alarms,
                sleepiness_scores[("imagined_sentence", "pre")],
                sleepiness_scores[("imagined_sentence", "post")],
                sleepiness_scores[("imagined_sentence", "post")] - sleepiness_scores[("imagined_sentence", "pre")],
            ]
    
    questionnaire_scores = score_subject_questionnaires(subject_folder_path)
    subject_data_row = [subject_number] + full_sentence_data + imagined_sentence_data + questionnaire_scores

    return subject_data_row


def run_one_source(datasource):

    # Current directory and raw data folder
    cur_dir = Path(__file__).parent.resolve() 
    raw_data_dir = cur_dir / ".." / "raw_data" / datasource

    # Iterate over all of the subjects and collect their data
    output_data_rows = []
    for subject_folder_path in raw_data_dir.iterdir():

        # Skip if not a folder 
        if not subject_folder_path.is_dir():
            print(f"[Skipped] Skipped {subject_folder_path.name}: Not a directory")
        
        # Get subject's data and add to the list of subject data rows
        subject_data_row = run_one_subject(subject_folder_path)
        output_data_rows.append(subject_data_row)

    
    save_folder = cur_dir / "outputs"
    save_folder.mkdir(parents = True, exist_ok = True)
    with open(save_folder / f"{datasource.lower()}_data.csv", mode = "w", newline = '') as f:
        writer = csv.writer(f)
        header = [
            "subject_number",
            "fs_block_order",
            "fs_total_targets",
            "fs_total_distractors",
            "fs_hits",
            "fs_falsealarms",
            "fs_sleepiness_pre",
            "fs_sleepiness_post",
            "fs_sleepiness_delta",
            "is_block_order",
            "is_total_targets",
            "is_total_distractors",
            "is_hits",
            "is_falsealarms",
            "is_sleepiness_pre",
            "is_sleepiness_post",
            "is_sleepiness_delta",
            "flow_state_scale_raw_sum",
            "tellegen_raw_sum",
            "vhq_raw_sum",
            "launay_slade_raw_sum",
            "dissociative_experiences_raw_sum",
            "bais_v_raw_sum",
            "bais_c_raw_sum",
            "flow_state_scale_mean",
            "tellegen_mean",
            "vhq_yes_proportion",
            "launay_slade_mean",
            "des_mean_0_100",
            "bais_v_mean",
            "bais_c_mean",
        ]

        writer.writerow(header)
        writer.writerows(output_data_rows)


def main():

    # Places we pull the data from
    datasources = ["Oberlin", "UChicago"]

    # Run each of the datasources
    for datasource in datasources:
        run_one_source(datasource)    


if __name__ == "__main__":
    main()