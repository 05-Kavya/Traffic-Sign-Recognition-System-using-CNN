import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model


model = load_model("traffic_sign_model.keras")



# Traffic sign class names


classes = [
    "Speed Limit 20 km/h",
    "Speed Limit 30 km/h",
    "Speed Limit 50 km/h",
    "Speed Limit 60 km/h",
    "Speed Limit 70 km/h",
    "Speed Limit 80 km/h",
    "End of Speed Limit 80 km/h",
    "Speed Limit 100 km/h",
    "Speed Limit 120 km/h",
    "No Passing",
    "No Passing - Trucks",
    "Right of Way",
    "Priority Road",
    "Yield",
    "STOP",
    "No Vehicles",
    "Trucks Prohibited",
    "No Entry",
    "General Caution",
    "Dangerous Curve Left",
    "Dangerous Curve Right",
    "Double Curve",
    "Bumpy Road",
    "Slippery Road",
    "Road Narrows",
    "Road Work",
    "Traffic Signals",
    "Pedestrians",
    "Children Crossing",
    "Bicycles Crossing",
    "Ice/Snow",
    "Wild Animals",
    "End of Speed/Passing Limits",
    "Turn Right Ahead",
    "Turn Left Ahead",
    "Ahead Only",
    "Straight or Right",
    "Straight or Left",
    "Keep Right",
    "Keep Left",
    "Roundabout",
    "End of No Passing",
    "End of No Passing - Trucks"
]



# Start webcam


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Traffic Sign Recognition started!")
print("Press Q to quit.")


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror the camera
    frame = cv2.flip(frame, 1)

    # Convert BGR → HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect red color
   

    lower_red1 = np.array([0, 80, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 80, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_box = None
    best_area = 0

   
    # Find largest red region
   

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore extremely thin regions
        if w < 30 or h < 30:
            continue

        # Prefer roughly square regions
        ratio = w / float(h)

        if 0.5 <= ratio <= 1.8:

            if area > best_area:
                best_area = area
                best_box = (x, y, w, h)


    
    # Classify detected sign
   

    if best_box is not None:

        x, y, w, h = best_box

        # Add a small margin
        margin = 10

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        sign = frame[y1:y2, x1:x2]

        if sign.size > 0:

            # Resize for CNN
            sign_resized = cv2.resize(sign, (32, 32))

            sign_resized = sign_resized.astype("float32") / 255.0

            sign_input = np.expand_dims(sign_resized, axis=0)

            # Prediction
            prediction = model.predict(
                sign_input,
                verbose=0
            )

            class_id = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # Display result
            label = f"{classes[class_id]}"

            confidence_text = f"Confidence: {confidence:.1f}%"

            cv2.putText(
                frame,
                label,
                (x1, max(30, y1 - 35)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                confidence_text,
                (x1, max(55, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


    
    # Display webcam
    

    cv2.imshow(
        "Traffic Sign Recognition",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()