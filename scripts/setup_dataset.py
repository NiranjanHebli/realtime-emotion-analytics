import kagglehub
import os
import shutil

def main():
    print("Downloading dataset using kagglehub...")
    # Download latest version of the human emotions dataset
    path = kagglehub.dataset_download("muhammadhananasghar/human-emotions-datasethes")
    print(f"Dataset downloaded to: {path}")
    
    # Path to the specific split folder within the downloaded Kaggle data
    source_dir = os.path.join(path, "EmotionsDataset_Splitted", "data")
    
    # The 'data' folder in our project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(project_root, "data")
    
    if os.path.exists(target_dir):
        print(f"Removing existing {target_dir}...")
        shutil.rmtree(target_dir)
        
    print(f"Copying files from {source_dir} to {target_dir}...")
    shutil.copytree(source_dir, target_dir)
    
    # Rename 'nothing' to 'neutral' to align with standard ML conventions
    for split in ['train', 'test']:
        nothing_path = os.path.join(target_dir, split, 'nothing')
        neutral_path = os.path.join(target_dir, split, 'neutral')
        if os.path.exists(nothing_path):
            print(f"Renaming '{nothing_path}' to '{neutral_path}'...")
            os.rename(nothing_path, neutral_path)
            
    print("Data successfully organized into 'data/train' and 'data/test'!")

if __name__ == "__main__":
    main()
