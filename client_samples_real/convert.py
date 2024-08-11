from PIL import Image
import os

def convert_jpg_to_png():
    # Get the list of files in the current directory
    files = os.listdir()

    # Filter out only JPG files
    jpg_files = [file for file in files if file.lower().endswith('.jpg')]

    # Convert each JPG file to PNG
    for jpg_file in jpg_files:
        jpg_path = os.path.join(os.getcwd(), jpg_file)
        png_file = os.path.splitext(jpg_file)[0] + '.png'
        png_path = os.path.join(os.getcwd(), png_file)

        # Open the JPG file and save it as PNG
        img = Image.open(jpg_path)
        img.save(png_path, 'PNG')

if __name__ == "__main__":
    # Call the function to convert JPG to PNG in the current directory
    convert_jpg_to_png()
