import os

base_path = "Dataset"
splits = ["train", "valid", "test"]

valid_ext = [".jpg", ".jpeg", ".png"]

for split in splits:
    images_dir = os.path.join(base_path, "images", split)
    labels_dir = os.path.join(base_path, "labels", split)

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ])

    print(f"{split} total images:", len(image_files))

    for idx, filename in enumerate(image_files):
        if filename.startswith(f"{split}_"):
            continue
        
        name, ext = os.path.splitext(filename)

        old_image_path = os.path.join(images_dir, filename)
        old_label_path = os.path.join(labels_dir, name + ".txt")

        if not os.path.exists(old_label_path):
            print(f"Warning: label tidak ditemukan untuk {filename}")
            continue

        new_name = f"{split}_{idx:04d}"

        new_image_path = os.path.join(images_dir, new_name + ext)
        new_label_path = os.path.join(labels_dir, new_name + ".txt")

        os.rename(old_image_path, new_image_path)
        os.rename(old_label_path, new_label_path)

print("Rename selesai.")


train_images_dir = os.path.join(base_path, "images", "train")
train_txt_path = os.path.join(base_path, "train.txt")

image_files = sorted([
    f for f in os.listdir(train_images_dir)
    if os.path.splitext(f)[1].lower() in valid_ext
])
with open(train_txt_path, "w") as f:
    for filename in image_files:
        f.write(f"images/train/{filename}\n")

print("train.txt berhasil diperbarui.")
    
for split in splits:
    images = set([os.path.splitext(f)[0] for f in os.listdir(os.path.join(base_path, "images", split))])
    labels = set([os.path.splitext(f)[0] for f in os.listdir(os.path.join(base_path, "labels", split))])

    print(split, "Mismatch:", images.symmetric_difference(labels))

