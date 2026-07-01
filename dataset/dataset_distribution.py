import os

classes = ["glioma", "meningioma", "notumor", "pituitary"]
test_folder = "Testing"

counts = {}
total = 0

# Count images in each class
for cls in classes:
    folder = os.path.join(test_folder, cls)
    num_images = len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    counts[cls] = num_images
    total += num_images

# Print percentage
print("\nDataset Distribution")
print("---------------------")

for cls in classes:
    percentage = (counts[cls] / total) * 100
    print(f"{cls}: {percentage:.2f}% ({counts[cls]} images)")

print(f"\nTotal Images: {total}")