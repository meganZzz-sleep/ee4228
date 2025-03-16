import os

# Define the path to your dataset
dataset_path = r'C:\Users\yelhs\source\repos\EE4228clone\Photos I havent sorted - ash'  # Replace this with the actual path

# Print the dataset path for confirmation
print(f"Dataset path: {dataset_path}")

# Loop through each folder (group member's folder)
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    
    if os.path.isdir(folder_path):  # Make sure it's a folder
        # Print the folder being processed
        print(f"Processing folder: {folder_name}")
        
        # List all files in the folder
        images = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Print the list of images found in the folder
        print(f"Images found: {images}")
        
        # Rename images sequentially if there are any images
        if images:
            for i, image_name in enumerate(images, start=1):
                # Construct the new image name
                new_name = f"img{i}.jpg"  # Change '.jpg' to '.png' if needed
                
                # Get the old and new file paths
                old_image_path = os.path.join(folder_path, image_name)
                new_image_path = os.path.join(folder_path, new_name)
                
                # Rename the file
                os.rename(old_image_path, new_image_path)
                
            print(f"Renamed images in folder {folder_name}")
        else:
            print(f"No images found in folder {folder_name}")