from pathlib import Path

def fix_labels_in_folder(folder_path):
    folder = Path(folder_path)
    label_files = list(folder.rglob("*.txt"))
    count = 0

    for label_file in label_files:
        lines = []
        fixed = False
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                try:
                    # Try to convert first part to float (valid YOLO)
                    float(parts[0])
                    lines.append(line.strip())  # Already fine
                except:
                    if len(parts) == 5:
                        new_line = "0 " + " ".join(parts[1:])
                        lines.append(new_line)
                        fixed = True
        if fixed:
            with open(label_file, 'w') as f:
                for l in lines:
                    f.write(l + '\n')
            count += 1

    print(f"✅ Fixed {count} label files in {folder_path}")

# Run the fix on your annotation folder
fix_labels_in_folder("archive/dataset/annotations")
fix_labels_in_folder("yolo_dataset/labels/train")
fix_labels_in_folder("yolo_dataset/labels/val")
