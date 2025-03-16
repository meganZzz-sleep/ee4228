import cv2
import os

def crop_images(input_folder, output_folder, size=(480, 480)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to greyscale
            height, width = img.shape[:2]
            start_x = max(0, (width - size[0]) // 2)
            start_y = max(0, (height - size[1]) // 2)
            cropped_img = img[start_y:start_y + size[1], start_x:start_x + size[0]]
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped_img)
            print(f"Cropped, converted to greyscale, and saved: {output_path}")
        else:
            print(f"Skipping invalid image: {img_path}")

# Example usage
input_folder = "input"  # Replace with your folder path
output_folder = "output"  # Replace with your folder path
crop_images(input_folder, output_folder)
