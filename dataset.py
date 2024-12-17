import os

# Define the base directory and subfolders
base_path = 'data'
subfolders = ['train', 'val']
food_classes = ['burger', 'pizza', 'fries']

# Create the directory structure
for subfolder in subfolders:
    for food_class in food_classes:
        os.makedirs(os.path.join(base_path, subfolder, food_class), exist_ok=True)

print("Directory structure created successfully.")
