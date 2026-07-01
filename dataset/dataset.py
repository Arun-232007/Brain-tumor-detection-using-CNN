import os

print("========== STEP 1: Dataset Distribution ==========")
os.system("python dataset_distribution.py")

print("\n========== STEP 2: Train Model ==========")
os.system("python brain_tumor.py")

print("\n========== STEP 3: Predict ==========")
os.system("python predict.py")

print("\nProject completed successfully!")