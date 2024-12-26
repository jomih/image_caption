#Firstly, you send a HTTP request to the provided URL and retrieve the webpage's content. This content is then parsed 
# by BeautifulSoup, which creates a parse tree from page's HTML. You look for 'img' tags in this parse tree as they contain 
# the links to the images hosted on the webpage.

import glob
import os
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, BlipForConditionalGeneration

os.environ['CURL_CA_BUNDLE'] = '/Users/jomih/Documents/90.Cajon Desastre/Coursera/02.IA_developer_by_IBM/01.Projects/07.Image_Caption/huggingface.co.pem'

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Specify the directory where your images are
image_dir = "./Fotos/"
image_exts = ["JPG", "jpg", "jpeg", "png"]  # specify the image file extensions to search for

# Open a file to write the captions
with open("captions.txt", "w") as caption_file:
    # Iterate over each img element
    for img_element in image_exts:
        for img_path in glob.glob(os.path.join(image_dir, f"*.{img_element}")):
            print(f"Image path is {img_path}")
            try:

                # Convert the image data to a PIL Image
                raw_image = Image.open(img_path).convert('RGB')

                if raw_image.size[0] * raw_image.size[1] < 400:  # Skip very small images
                    print(f"Skip image {img_path}")
                    continue
            
                raw_image = raw_image.convert('RGB')

                # Process the image
                inputs = processor(raw_image, return_tensors="pt")

                # Generate a caption for the image
                out = model.generate(**inputs, max_new_tokens=50)

                # Decode the generated tokens to text
                caption = processor.decode(out[0], skip_special_tokens=True)

                # Write the caption to the file, prepended by the image URL
                caption_file.write(f"{img_path}: {caption}\n")
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue