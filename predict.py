import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("traffic_sign_model.keras") # 1. Load trained model

classes = [        # 2. Traffic sign class names
    "Speed limit 20 km/h",
    "Speed limit 30 km/h",
    "Speed limit 50 km/h",
    "Speed limit 60 km/h",
    "Speed limit 70 km/h",
    "Speed limit 80 km/h",
    "End of speed limit 80 km/h",
    "Speed limit 100 km/h",
    "Speed limit 120 km/h",
    "No passing",
    "No passing for vehicles over 3.5 tons",
    "Right-of-way at intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 tons"
]

image_path = input("Enter the path of the traffic sign image: ")    #Enter image path

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not load image.")
    exit()




image_resized = cv2.resize(image, (32, 32)) # Preparing image

image_normalized = image_resized.astype("float32") / 255.0

image_input = np.expand_dims(image_normalized, axis=0)



# Prediction

prediction = model.predict(image_input, verbose=0)

class_id = np.argmax(prediction)
confidence = np.max(prediction) * 100


# Displaying result


print("\n----------------------------")
print("TRAFFIC SIGN RECOGNITION")
print("----------------------------")

print("Predicted Class:", class_id)
print("Traffic Sign:", classes[class_id])
print(f"Confidence: {confidence:.2f}%")

print("----------------------------")