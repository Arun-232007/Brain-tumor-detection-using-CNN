import os
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1. Debug folder
print("Current folder:", os.getcwd())
print("Files here:", os.listdir())

# 2. Load model
model = load_model("brain_tumor_model.h5")

# 3. Load image (CHANGE path)
img_path = "test.jpg"

img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

# 4. Predict
predictions = model.predict(img)

# 5. Class names (EDIT THIS)
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

for i, class_name in enumerate(class_names):
    print(f"{class_name}: {predictions[0][i]*100:.2f}%")