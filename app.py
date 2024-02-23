from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__, static_url_path='/static')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Variables to store the last computed values
last_mean_intensity = None
last_product_recommendation = None
last_product_image = None

# Flag to indicate whether the analysis should continue
analysis_continues = True

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return None
    return faces[0]

def analyze_pigmentation(face_roi):
    # Convert face_roi to LAB color space
    lab_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2Lab)

    # Extract the L channel (luminance) which represents lightness
    l_channel = lab_roi[:, :, 0]

    # Calculate the mean intensity of the L channel
    mean_intensity = np.mean(l_channel)

    return mean_intensity

def recommend_product(mean_intensity):
    # Actual recommendation logic based on pigmentation
    # Replace this with your own logic

    if mean_intensity < 100:
        return "Brightening Cream for Very Light Pigmentation", "/static/brightening_cream_image.jpg"
    elif 100 <= mean_intensity < 130:
        return "Moisturizer for Normal Pigmentation", "/static/moisturizer_image.jpg"
    elif 130 <= mean_intensity < 160:
        return "Sunscreen for Slight Dark Pigmentation", "/static/sunscreen_image.jpg"
    else:
        return "Serum for Intense Dark Pigmentation", "/static/serum_image.jpg"

def generate_frames():
    global analysis_continues, last_mean_intensity, last_product_recommendation, last_product_image
    cap = cv2.VideoCapture(0)

    frame_bytes = b''  # Initialize frame_bytes outside the loop

    while analysis_continues:
        ret, frame = cap.read()

        if not ret:
            break

        face_coords = detect_face(frame)

        if face_coords is not None:
            x, y, w, h = face_coords
            face_roi = frame[y:y+h, x:x+w]

            mean_intensity = analyze_pigmentation(face_roi)

            cv2.putText(frame, f"Mean Intensity: {mean_intensity:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Actual recommendation logic
            product_recommendation, product_image = recommend_product(mean_intensity)

            last_mean_intensity = mean_intensity
            last_product_recommendation = product_recommendation
            last_product_image = product_image

            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            recommendation_script = f'<script>updateResult("{mean_intensity:.2f}", "{product_recommendation}", "{product_image}");</script>\r\n'
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n' +
                   recommendation_script.encode('utf-8'))

        else:
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    # Release the video capture object
    cap.release()

    # After analysis is stopped, yield the last computed values
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    last_mean_intensity = None
    last_product_recommendation = None
    last_product_image = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to stop analysis
@app.route('/stop_analysis')
def stop_analysis():
    global analysis_continues, last_mean_intensity, last_product_recommendation, last_product_image
    analysis_continues = False

    # Actual pigmentation analysis and product recommendation
    mean_intensity = last_mean_intensity if last_mean_intensity is not None else np.random.uniform(80, 150)
    product_recommendation = last_product_recommendation if last_product_recommendation is not None else "Product Recommendation Not Available"
    product_image = last_product_image if last_product_image is not None else "/static/default_image.jpg"

    return {
        "mean_intensity": mean_intensity,
        "product_recommendation": product_recommendation,
        "product_image": product_image
    }

if __name__ == "__main__":
    app.run(debug=True)
