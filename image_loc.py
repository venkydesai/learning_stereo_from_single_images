import os

def save_image_locations(folder_path, output_file):
    with open(output_file, 'w') as file:
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                image_2_path = os.path.join('testing', 'image_2', filename)
                image_3_path = os.path.join('testing', 'image_3', filename)
                line = f"{image_2_path} {image_3_path}\n"
                file.write(line)

if __name__ == "__main__":
    # Replace 'your_folder_path' with the path to the folder containing the images
    folder_path = 'testing/image_2'


    # Replace 'output.txt' with the desired name of the output text file
    output_file = 'testing_file.txt'

    save_image_locations(folder_path, output_file)
