import os
import sys

# append parent
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append( parent_dir)
sys.path.append( os.path.join(parent_dir, 'src') )

from PIL import Image

from torchvision import transforms
import constants

# Only valid files can be uploaded
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_uploaded_files(upload_folder):
    files = []
    if os.path.exists(upload_folder):
        for filename in os.listdir(upload_folder):
            path = os.path.join(upload_folder, filename)
            size = os.path.getsize(path)
            modified_time = os.path.getmtime(path)
            files.append({
                "name": filename,
                "size": size,
                "modified": modified_time
            })
    return files

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(constants.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image)
