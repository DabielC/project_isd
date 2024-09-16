import io, base64
from PIL import Image
image = 
# Assuming base64_str is the string value without 'data:image/jpeg;base64,'
img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
img.save('my-image.jpeg')