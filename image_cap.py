import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image, DONT FORGET TO WRITE YOUR IMAGE NAME
img_path = "/Users/jomih/Documents/20.Juniper/100.Scripts/blip/6a862d52-8dc4-471d-b75f-31cc65b0e7da.jpg"

# convert it into an RGB format 
image = Image.open(img_path).convert('RGB')

# You do not need a question for image captioning
# Next, the pre-processed image is passed through the processor to generate inputs in the required format.
# The return_tensors argument is set to "pt" to return PyTorch tensors
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")

# Generate a caption for the image, with up to 50 tokens in length
#The two asterisks (**) in Python are used in function calls to unpack dictionaries and pass items in the dictionary as 
# keyword arguments to the function. **inputs is unpacking the inputs dictionary and passing its items as arguments to the model.
outputs = model.generate(**inputs, max_length=50)

# Decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)

# Print the caption
print(caption)

#run the project with python3 image_cap.py