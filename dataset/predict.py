import os
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image

print("Loading model...")
model = load_model("brain_tumor_model.h5")
print("Model loaded successfully!")

class_names = ["glioma", "meningioma", "notumor", "pituitary"]
test_folder = "Testing"

total_probs = np.zeros(4)
total_images = 0

for folder in class_names:
    folder_path = os.path.join(test_folder, folder)
    print(f"\nChecking folder: {folder_path}")

    if not os.path.exists(folder_path):
        print("Folder not found!")
        continue

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"Found {len(files)} images")

    for file in files:
        img_path = os.path.join(folder_path, file)

        img = image.load_img(img_path, target_size=(128, 128))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img, verbose=0)

        total_probs += predictions[0]
        total_images += 1

print("\nTotal images processed:", total_images)

if total_images > 0:
    average_probs = (total_probs / total_images) * 100

    print("\n===== Overall Prediction Percentage =====")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {average_probs[i]:.2f}%")
else:
    print("No images were processed.")