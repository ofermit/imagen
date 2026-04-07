import glob
from collections import Counter
from PIL import Image

def main():
    images = glob.glob("data/images/*.jpg")
    sizes = Counter()
    for img_path in images:
        with Image.open(img_path) as img:
            sizes[img.size] += 1
            
    print("Image resolutions found in data/images:")
    for size, count in sizes.most_common(10):
        print(f"  {size[0]}x{size[1]}: {count} images")

if __name__ == "__main__":
    main()
