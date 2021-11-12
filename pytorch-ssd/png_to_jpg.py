from PIL import Image
import os
import sys

# python png_to_jpg.py data/helmet/
def main():
    global directory
    os.chdir(directory)
    print(os.getcwd())
    for filename in os.listdir("./JPEGImages"):
        if filename.endswith(".png"):
            img = Image.open(os.getcwd() + '/JPEGImages/' + filename,mode='r')
            filename = (os.getcwd() + '/JPEGImages/' + filename).rstrip('png')
            img.save(filename + 'jpg')

if __name__ == '__main__':
    directory = sys.argv[1]
    main()
