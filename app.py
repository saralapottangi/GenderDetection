from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained models
def load_models():
    # Load face detection model
    face_proto = "models/opencv_face_detector.pbtxt"
    face_model = "models/opencv_face_detector_uint8.pb"
    face_net = cv2.dnn.readNet(face_model, face_proto)
    
    # Load age detection model
    age_proto = "models/age_deploy.prototxt"
    age_model = "models/age_net.caffemodel"
    age_net = cv2.dnn.readNet(age_model, age_proto)
    
    # Load gender detection model
    gender_proto = "models/gender_deploy.prototxt"
    gender_model = "models/gender_net.caffemodel"
    gender_net = cv2.dnn.readNet(gender_model, gender_proto)
    
    return face_net, age_net, gender_net

face_net, age_net, gender_net = load_models()

# Model parameters
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

def get_face_box(net, frame, conf_threshold=0.7):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    
    net.setInput(blob)
    detections = net.forward()
    
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
    return face_boxes

def detect_age_gender(frame):
    # Create a copy of the frame to draw on
    result_img = frame.copy()
    
    # Get face box
    face_boxes = get_face_box(face_net, frame)
    
    results = []
    
    for face_box in face_boxes:
        face = frame[max(0, face_box[1]):min(face_box[3], frame.shape[0]-1), 
                    max(0, face_box[0]):min(face_box[2], frame.shape[1]-1)]
        
        # Extract face
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Gender detection
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        
        # Age detection
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        
        # Draw rectangle around face
        cv2.rectangle(result_img, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 255, 0), 2)
        
        # Label with age and gender
        label = f"{gender}, {age}"
        cv2.putText(result_img, label, (face_box[0], face_box[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        results.append({
            "gender": gender,
            "age": age,
            "face_box": face_box
        })
    
    return result_img, results

# Global variable for webcam
camera = None

def generate_frames():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            result_img, _ = detect_age_gender(frame)
            ret, buffer = cv2.imencode('.jpg', result_img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({"status": "success"})

@app.route('/predict_from_image', methods=['POST'])
def predict_from_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Read image
        frame = cv2.imread(file_path)
        result_img, results = detect_age_gender(frame)
        
        # Save processed image
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], "result_" + filename)
        cv2.imwrite(result_path, result_img)
        
        return jsonify({
            "results": results,
            "original_image": file_path,
            "processed_image": result_path
        })

if __name__ == "__main__":
    app.run(debug=True)