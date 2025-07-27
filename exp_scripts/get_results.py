import os
import json
import re
import pandas as pd

def find_folders_with_keyword(root_dir, keywords):
    """
    Recursively find all folders containing a specific keyword in their names 
    within a given directory.

    :param root_dir: The root directory to start the search from.
    :param keywords: A list of keywords to search for in folder names.
    :return: A list of paths to folders that contain the keyword in their names.
    """
    matching_folders = []

    # Walk through the directory and its subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if the current folder name contains the keyword
        flag = True
        assert type(keywords) is list
        for keyword in keywords:
            if keyword not in os.path.basename(dirpath):
                flag = False
        if flag:
            matching_folders.append(dirpath)

    return matching_folders


def find_metric_json_files(folder):
    """
    Recursively find all JSON files containing "metric_" in their names within 
    a given folder and its subdirectories.

    :param folder: The folder to search for JSON files.
    :return: A list of paths to JSON files that contain "metric_" in their 
        names.
    """
    metric_json_files = []

    # Walk through the folder and its subdirectories
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            # Check if the file is a JSON file and contains "metric_" in its name
            if filename.endswith(".json") and "metric_" in filename:
                metric_json_files.append(os.path.join(dirpath, filename))

    return metric_json_files


def flatten_json_fields(data):
    """
    Flatten a JSON structure by merging two-level fields into one-level fields.

    :param data: The JSON data (as a dictionary) to flatten.
    :return: A flattened dictionary with merged keys.
    """
    flattened_data = {}

    for parent_key, child_dict in data.items():
        if isinstance(child_dict, dict):  # Check if the value is a dictionary
            for child_key, value in child_dict.items():
                # Merge parent and child keys with an underscore
                merged_key = f"{parent_key}_{child_key}"
                flattened_data[merged_key] = value
        else:
            # If the value is not a dictionary, keep it as is
            flattened_data[parent_key] = child_dict

    return flattened_data


def extract_iteration_number(file_path):
    return int(os.path.basename(file_path).split("_")[-1].split(".")[0])


def extract_task_name(folder):
    return folder.split("/")[-1].split("_")[5]


def extract_metric_files(metric_files, fields_to_keep):
    """
    Extract specified fields from each JSON file.

    :param metric_files: A list of paths to metric JSON files.
    :param fields_to_keep: A list of fields to retain from each JSON file.
    :return: A list of dictionaries, where each dictionary contains the retained
        fields.
    """

    # Extract specified fields from each JSON file
    extracted_data = []
    for file_path in metric_files:
        with open(file_path, "r") as file:
            data = json.load(file)
            flattened_data = flatten_json_fields(data)
            # Retain only the specified fields
            filtered_data = {field: flattened_data.get(field) for field in fields_to_keep}
            extracted_data.append(filtered_data)

    return extracted_data


def select_iteration_result(extracted_data, criterion="last", metric_name=None):
    """
    Select a specific iteration's result from extracted_data based on a 
    criterion.

    :param extracted_data: A list of dictionaries, where each dictionary 
        contains the retained fields.
    :param criterion: The selection criterion. Options: "last" (last iteration) 
        or "best" (iteration with the highest metric score).
    :return: The selected iteration's result (a dictionary).
    """
    if criterion == "last":
        # Select the last iteration's result
        return extracted_data[-1]
    elif criterion == "best":
        # Select the iteration with the highest metric score
        assert metric_name is not None, f"'best' criterion needs `metric_name`."
        return max(extracted_data, key=lambda x: x.get(metric_name, float("-inf")))
    else:
        raise ValueError(f"Invalid criterion: {criterion}. Choose 'last' or 'best'.")


def aggregate_results_to_dataframe(folders, fields_to_keep, criterion, best_metric_name):
    """
    Aggregate selected results from all folders into a DataFrame.

    :param folders: A list of folder paths.
    :param fields_to_keep: A list of fields to retain from each JSON file.
    :param criterion: The selection criterion. Options: "last" or "best".
    :param best_metric_name: The metric name to use for the "best" criterion.
    :return: A DataFrame with tasks as columns and metrics as rows.
    """
    results = {}

    for folder in folders:
        # Find all "metric_*.json" files in the current folder
        metric_files = find_metric_json_files(folder)
        metric_files = sorted(metric_files, key=extract_iteration_number)
        assert metric_files, f"No 'metric_*.json' files found in '{folder}'."

        # Extract specified fields from each JSON file
        extracted_data = extract_metric_files(metric_files, fields_to_keep)

        # Select a specific iteration's result based on a criterion
        selected_result = select_iteration_result(extracted_data, 
                                                  criterion=criterion, 
                                                  metric_name=best_metric_name)

        # Extract task name
        task_name = extract_task_name(folder)

        # Store the selected result in the results dictionary
        results[task_name] = selected_result

    # Convert the results dictionary to a DataFrame
    df = pd.DataFrame(results)
    return df



if __name__ == "__main__":

    ##############  CONFIGS  ##############
    # Define the root directory and the keyword to search for
    root_directory = "/home/hewn/TASC/output"  # Replace with your target directory
    keywords = ["runs_", "_officehome_", "OPDA"]     # Replace with your desired keyword
    # Define the fields to retain from each JSON file
    fields_to_keep = ["Open-set_OS*", "Open-set_UNK", "Unknown_binary_AUROC", "UniDA_H-score"]  # Replace with your desired fields
    criterion = 'last'
    best_metric_name = 'UniDA_H-score' if criterion == 'best' else None
    #######################################


    # Find all folders containing the keyword
    folders = find_folders_with_keyword(root_directory, keywords)
    folders = sorted(folders, key=extract_task_name)

    # Output the results
    assert folders, f"No folders containing '{keywords}' were found."
    print(f"Found folders containing '{keywords}':")

    # Print metadata information
    print("\nExperiment Configuration:")
    print(f"Root Directory: {root_directory}")
    print(f"Keywords: {keywords}")
    print(f"Fields to Keep: {fields_to_keep}")
    print(f"Criterion: {criterion}")
    print(f"Best Metric Name: {best_metric_name}")
    print(f"Folders Processed: {len(folders)}")
    for folder in folders:
        print(f" - {folder}")

    # Aggregate results into a DataFrame
    df = aggregate_results_to_dataframe(folders, fields_to_keep, criterion, best_metric_name)
    print("Aggregated results:")

    # Add a column for the average of each row
    df['Avg'] = df.mean(axis=1)
    print(df)



