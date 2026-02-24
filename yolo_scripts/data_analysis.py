import os
import argparse
from collections import Counter

def analyze_labels(root_path):
    # Initialize counters
    overall_counts = Counter()
    folder_stats = {}

    print(f"{'='*60}")
    print(f"Analyzing YOLO dataset at: {root_path}")
    print(f"{'='*60}\n")

    # Walk through the directory tree
    for root, dirs, files in os.walk(root_path):
        # We only care about folders that contain labels
        if "labels" in root.lower():
            txt_files = [f for f in files if f.lower().endswith('.txt')]
            if not txt_files:
                continue

            current_folder_counts = Counter()
            
            for filename in txt_files:
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            parts = line.split()
                            if parts:
                                class_id = parts[0]
                                current_folder_counts[class_id] += 1
                                overall_counts[class_id] += 1
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

            folder_stats[root] = current_folder_counts

    # Mapping for display
    class_map = {'0': 'Person', '1': 'Car'}

    # Display results per subfolder
    print(f"{'Folder Path':<50} | {'Person (0)':<10} | {'Car (1)':<10}")
    print("-" * 75)
    
    for folder, counts in folder_stats.items():
        # Displaying folder name relative to root for clarity
        rel_folder = os.path.relpath(folder, root_path)
        p_count = counts.get('0', 0)
        c_count = counts.get('1', 0)
        print(f"{rel_folder:<50} | {p_count:<10} | {c_count:<10}")

    # Display Grand Total
    print(f"\n{'='*60}")
    print(f"{'GRAND TOTALS':<50}")
    print(f"{'='*60}")
    print(f"Total Persons (Class 0): {overall_counts.get('0', 0)}")
    print(f"Total Cars    (Class 1): {overall_counts.get('1', 0)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze class distributions in YOLO label folders.")
    parser.add_argument("--input", type=str, default="filtered_data", help="Path to the filtered_data folder")
    
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        analyze_labels(args.input)
    else:
        print(f"Error: Folder '{args.input}' not found.")