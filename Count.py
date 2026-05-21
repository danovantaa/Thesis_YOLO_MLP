import os
import csv

base_path = "Dataset/labels"

folders = {
    "train": os.path.join(base_path, "train"),
    "valid": os.path.join(base_path, "valid"),
    "test": os.path.join(base_path, "test")
}

UPPER_BODY_CLASS = 2

for dataset_name, folder_path in folders.items():

    data = []

    for label_file in sorted(os.listdir(folder_path)):

        if label_file.endswith(".txt"):

            path = os.path.join(folder_path, label_file)

            count = 0

            with open(path, "r") as f:
                for line in f:
                    class_id = int(line.split()[0])

                    if class_id == UPPER_BODY_CLASS:
                        count += 1

            image_name = label_file.replace(".txt", ".jpg")

            data.append([image_name, count])


    output_csv = f"{dataset_name}_count.csv"

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["NamaFile", "BanyakOrang"])
        writer.writerows(data)

    print(f"{output_csv} berhasil dibuat")