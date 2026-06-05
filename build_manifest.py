import os
import pandas as pd

def normalize_string(text):
    """Removes spaces, punctuation, and forces lowercase for robust matching."""
    if pd.isna(text):
        return ""
    return str(text).strip().lower().replace(" ", "").replace(".", "").replace(",", "").replace("-", "")

def build_robust_manifest(dataset_root="Merged_EthnoFace_Dataset_cleaned_on_25May", output_csv="multitask_manifest.csv"):
    all_records = []
    target_csv_name = "metadata_thesis_unique.csv"
    
    if not os.path.exists(dataset_root):
        print(f"Error: The root folder '{dataset_root}' does not exist in the current directory.")
        return

    print("=" * 80)
    print(f"Building Clean Multi-Task Manifest tracking: {target_csv_name}")
    print("=" * 80)

    # Map out actual folder names on disk to resolve formatting typos
    folder_map = {normalize_string(f): f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))}

    for country_folder in os.listdir(dataset_root):
        country_path = os.path.join(dataset_root, country_folder)
        
        if os.path.isdir(country_path):
            local_csv_path = os.path.join(country_path, target_csv_name)
            if not os.path.exists(local_csv_path):
                continue
                
            try:
                df = pd.read_csv(local_csv_path, header=0)
                
                # Verify that required columns exist
                required = ['filename', 'country', 'age', 'gender', 'ethnicity']
                if not all(col in df.columns for col in required):
                    print(f"⚠️ Skipping '{country_folder}': Missing header columns.")
                    continue

                rows_added = 0
                for _, row in df.iterrows():
                    if pd.isna(row['filename']) or pd.isna(row['country']):
                        continue
                        
                    img_name = str(row['filename']).strip()
                    csv_country = str(row['country']).strip()
                    
                    # Correct matching variations (handles Virgin Islands, typos, etc.)
                    corrected_folder = folder_map.get(normalize_string(csv_country), country_folder)
                    corrected_path = os.path.join(dataset_root, corrected_folder)

                    # Look for image files inside an 'images' folder or directly in the country root
                    possible_img_path = os.path.join(corrected_path, "images", img_name)
                    if not os.path.exists(possible_img_path):
                        possible_img_path = os.path.join(corrected_path, img_name)
                        
                    if os.path.exists(possible_img_path):
                        all_records.append({
                            'image_path': possible_img_path,
                            'country': corrected_folder,
                            'age': str(row['age']).strip() if not pd.isna(row['age']) else 'unknown',
                            'gender': str(row['gender']).strip() if not pd.isna(row['gender']) else 'unknown',
                            'ethnicity': str(row['ethnicity']).strip() if not pd.isna(row['ethnicity']) else 'unknown'
                        })
                        rows_added += 1
                        
                if rows_added > 0:
                    print(f"✅ Indexed {rows_added} valid records from {country_folder}")
                    
            except Exception as e:
                print(f"❌ Error reading {local_csv_path}: {e}")

    if all_records:
        pd.DataFrame(all_records).to_csv(output_csv, index=False)
        print("\n" + "=" * 80)
        print(f"🎉 SUCCESS: Generated '{output_csv}' with {len(all_records)} valid rows!")
        print("=" * 80)
    else:
        print("❌ Error: No matching image paths found on disk.")

if __name__ == "__main__":
    build_robust_manifest()