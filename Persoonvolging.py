import cv2
from ultralytics import YOLO
import math
import serial
import time
from flask import Flask, Response, render_template_string, request, jsonify
# Serial starten
ser = serial.Serial("COM8", 9600, timeout=1)
time.sleep(2)
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
circle_radius = 40
x_distance = 70
max_area = 200000
app = Flask(__name__)
auto_mode = True
# HTML met real-time knoppenbediening
html_template = """
<html>
<head>
<title>Robot Camera & Control</title>
<style>
button { width: 120px; height: 50px; font-size: 16px; margin: 5px; }
</style>
<script>
function sendCommand(cmd){
    fetch("/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ "command": cmd })
    });
}
</script>
</head>
<body>
<h1>Live Camera Feed</h1>
<img src="/video_feed" width="800" /><br>
<form action="/toggle_auto" method="post">
    <button type="submit">{{ 'Stop Auto' if auto_mode else 'Start Auto' }}</button>
</form>
{% if not auto_mode %}
<div>
    <button onmousedown="sendCommand('v 120 0')" onmouseup="sendCommand('s 120 0')">Vooruit</button>
    <button onmousedown="sendCommand('a 120 0')" onmouseup="sendCommand('s 120 
        0')">Achteruit</button><br>
    <button onmousedown="sendCommand('q 120 0')" onmouseup="sendCommand('s 120 0')">Links</button>
    <button onmousedown="sendCommand('d 120 0')" onmouseup="sendCommand('s 120 0')">Rechts</button>
</div>
{% endif %}
</body>
</html>
"""
def generate_frames():
    global auto_mode
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if auto_mode:
            results = model(frame)
            best_person = None
            best_conf = 0
            height, width, _ = frame.shape
            frame_center_x = width // 2
            frame_center_y = height // 2
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf > best_conf:
                    best_conf = conf
                    best_person = box
            positie = ""
            turn = False
            if best_person is not None:
                x1, y1, x2, y2 = map(int, best_person.xyxy[0])
                width_box = x2 - x1
                height_box = y2 - y1
                area = width_box * height_box
                person_center_x = (x1 + x2) // 2
                person_center_y = (y1 + y2) // 2
                if person_center_x < frame_center_x - x_distance:
                    positie = "r 100 50"
                    turn = True
                elif person_center_x > frame_center_x + x_distance:
                    positie = "l 100 50"
                    turn = True
                if area < max_area and not turn:
                    positie = "f 120 50"
                if positie != "":
                    ser.write((positie + "\n").encode())
                # Visuals
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(frame, (frame_center_x, frame_center_y),
                         (person_center_x, person_center_y), (0, 0, 255), 2)
        # Middenlijn en cirkels
        height, width, _ = frame.shape
        frame_center_x = width // 2
        frame_center_y = height // 2
        cv2.line(frame, (frame_center_x, 0), (frame_center_x, height), (255, 0, 0), 2)
        cv2.circle(frame, (frame_center_x, frame_center_y), 5, (255, 255, 255), -1)
        cv2.circle(frame, (frame_center_x, frame_center_y), circle_radius, (255, 255, 255), 2)
        cv2.line(frame, (frame_center_x - x_distance, 0), (frame_center_x - x_distance, height), (255, 0,
             0), 2)
        cv2.line(frame, (frame_center_x + x_distance, 0), (frame_center_x + x_distance, height), (255, 0,
             0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/')
def index():
    return render_template_string(html_template, auto_mode=auto_mode)
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/toggle_auto', methods=['POST'])
def toggle_auto():
    global auto_mode
    auto_mode = not auto_mode
    return render_template_string(html_template, auto_mode=auto_mode)
@app.route('/control', methods=['POST'])
def control():
    data = request.get_json()
    cmd = data.get("command")
    if cmd:
        ser.write((cmd + "\n").encode())
    return jsonify(success=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
