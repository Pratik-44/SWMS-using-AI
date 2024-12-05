import os
import pickle
import sys
import time
import cv2
import face_recognition
import numpy as np
import pandas as pd
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for,flash
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime as dt_class
import datetime as dt_module
import humanize
from pygame import mixer
import queue
import threading
from scipy.spatial import distance
import dlib
from imutils import face_utils
import imutils
import pywhatkit as kit
import logging
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

mixer.init()
mixer_queue = queue.Queue()

# firebase initialization
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://swms-ced78-default-rtdb.firebaseio.com/',
    'storageBucket': 'swms-ced78.appspot.com'
})

# Global declaration
camera = cv2.VideoCapture(0)
file_path = os.path.join('static', 'model', 'EncoderFile.p')
file = open(file_path, 'rb')
encodeListKnownWithIds = pickle.load(file)
encodedFaceKnown, employee_ids = encodeListKnownWithIds

local_images_folder = 'static/Images'
firebase_images_folder = 'Images'

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/Images'

# Add a global variable to track admin login status
is_admin_logged_in = False


# Add a login route to handle user authentication
@app.route('/login', methods=['GET', 'POST'])
def login():
    global is_admin_logged_in

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if username and password match admin credentials
        if username == 'Admin' and password == 'Admin@1234':
            is_admin_logged_in = True
            success_message = 'Login successful'
            return redirect(url_for('index', success_message=success_message))  # Redirect to home page after successful login
        else:
            flash('Invalid username or password', 'error')  # Show error message for invalid credentials

    return render_template('login.html')

# Add a logout route to logout the admin
@app.route('/logout')
def logout():
    global is_admin_logged_in
    is_admin_logged_in = False
    return redirect(url_for('index')) # Redirect to login page after logout

def encode_images(images_folder, model_folder):
    global encodedFaceKnown, employee_ids  # Declare global variables
    image_list = os.listdir(images_folder)
    encodings = []
    employee_ids = []

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    for image_name in image_list:
        image_path = os.path.join(images_folder, image_name)
        image = cv2.imread(image_path)
        employee_id_name = os.path.splitext(image_name)[0]
        employee_ids.append(employee_id_name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(image_rgb)[0]
        encodings.append(encoding)

    encodedFaceKnown = encodings
    encode_list_with_ids = [encodings, employee_ids]

    encoder_file_path = os.path.join(model_folder, "EncoderFile.p")
    with open(encoder_file_path, 'wb') as file:
        pickle.dump(encode_list_with_ids, file)

    print(f"Encoding Complete...")


def detect_faces():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            imgSmall = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(imgSmall)
            encode_current_frame = face_recognition.face_encodings(imgSmall, face_locations)

            if face_locations:
                for encode_face, face_location in zip(encode_current_frame, face_locations):
                    matches = face_recognition.compare_faces(encodedFaceKnown, encode_face, tolerance=0.4)
                    face_distance = face_recognition.face_distance(encodedFaceKnown, encode_face)

                    match_index = np.argmin(face_distance)

                    if matches[match_index]:
                        id = employee_ids[match_index]
                        yield id


def generate_frames():
    global encodedFaceKnown, camera
    detection_interval = 5  # Detection interval in seconds
    last_detection_time = time.time()

    while True:
        current_time = time.time()
        elapsed_time = current_time - last_detection_time

        success, frame = camera.read()
        if not success:
            break
        else:
            if elapsed_time >= detection_interval:
                imgSmall = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
                imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

                faceCurrentFrame = face_recognition.face_locations(imgSmall)
                encodeCurrentFrame = face_recognition.face_encodings(imgSmall, faceCurrentFrame)

                if faceCurrentFrame:
                    for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
                        matches = face_recognition.compare_faces(encodedFaceKnown, encodeFace, tolerance=0.5)
                        faceDistance = face_recognition.face_distance(encodedFaceKnown, encodeFace)

                        matchIndex = np.argmin(faceDistance)

                        if matches[matchIndex]:
                            id = employee_ids[matchIndex]
                            mark_attendance(id)

                last_detection_time = current_time

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/submit-form', methods=['POST'])
def submit_form():
    global is_admin_logged_in

    if not is_admin_logged_in:
        return redirect(url_for('login'))  # Redirect to login page if not logged in as admin

    if request.method == 'POST':
        # Get form data
        name = request.form['full-name']
        dob = request.form['dob']
        phone = request.form['phone']
        email = request.form['email']
        address = request.form['address']
        distance_from_home = request.form['distance-from-home']
        total_experience = request.form['total-experience']
        role = request.form['role']
        joining_date = request.form['joining-date']
        employee_id = request.form['id']
        password = request.form['password']

        # Check if employee_id exists in the form data
        if 'id' not in request.form:
            return 'Employee ID not found in form data!'

        id = request.form['id']

        # Prepare the data to be updated in Firebase
        data = {
            'name': name,
            'dob': dob,
            'phone': phone,
            'email': email,
            'address': address,
            'distance_from_home': distance_from_home,
            'total_experience': total_experience,
            'role': role,
            'joining_date': joining_date,
            'id': employee_id,
            'password': password,
            'total_attendance': 0,
            'last_attendance_time': 0,
        }
        db.reference(f'Employee Attendance/{id}').set(data)

        surveillance_data = {
            'id': employee_id,
            'name': name,
            'penalty': {
                'total_penalty': 0,
                'date': {
                    dt_module.datetime.now().strftime("%d-%m-%Y"): 0  # Initialize today's date with 0 penalty
                }
            }
        }
        db.reference(f'Surveillance/{employee_id}').set(surveillance_data)

        try:
            # Check if the photo key exists in the request.files dictionary
            if 'photo' not in request.files:
                return 'No photo part in the form!'

            photo = request.files['photo']

            # Check if the file was not selected
            if photo.filename == '':
                return 'No selected photo!'

            employee_id = request.form.get('id')

            # Rename the photo to id.jpg
            photo.filename = f"{employee_id}.jpg"

            # Save the photo
            photo.save(os.path.join(app.config['UPLOAD_FOLDER'], photo.filename))
            upload_images_to_firebase(local_images_folder, firebase_images_folder)
            images_folder_path = 'static/Images'
            model_folder_path = 'static/model'
            encode_images(images_folder_path, model_folder_path)
            success_message = 'Form submitted successfully!'
            return render_template('services.html', success_message=success_message)

        except Exception as e:
            error_message = f"Error uploading photo: {str(e)}"
            return render_template('services.html', error_message=error_message)


def upload_images_to_firebase(local_folder_path, firebase_folder_path):
    for filename in os.listdir(local_folder_path):
        local_file_path = os.path.join(local_folder_path, filename)
        if os.path.isfile(local_file_path):
            bucket = storage.bucket()
            blob = bucket.blob(firebase_folder_path + '/' + filename)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {filename} to Firebase Storage")


def mark_attendance(id):
    now = dt_class.now()
    dt_string = now.strftime('%d-%B-%Y')
    folder_name = "Attendance"
    filename = os.path.join(folder_name, dt_string + '.csv')

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write('Id,Name,Time,Total Time\n')

    with open(filename, 'r') as f:
        my_data_list = f.readlines()

    name_list = [line.strip().split(',')[0] for line in my_data_list]

    # dt_string = now.strftime('%H:%M:%S')
    if id not in name_list:
        dt_string = now.strftime('%H:%M:%S')
        date = dt_class.now().strftime("%Y-%m-%d")
        employee_info = db.reference(f'Employee Attendance/{id}').get()
        ref = db.reference(f'Employee Attendance/{id}')
        name = ref.child('name').get()
        employee_info['total_attendance'] += 1
        employee_info['last_attendance_time'] = f"{date} {dt_string}"
        ref.child('total_attendance').set(employee_info['total_attendance'])
        ref.child('last_attendance_time').set(employee_info['last_attendance_time'])

        with open(filename, 'a') as f:
            f.write(f'{id},{name},{dt_string},0\n')
    else:
        index = name_list.index(id)
        info = my_data_list[index].strip().split(',')
        total_time = int(info[3]) + 5
        dt_string = info[2]
        name = info[1]
        my_data_list[index] = f'{id},{name},{dt_string},{total_time}\n'

        with open(filename, 'w') as f:
                f.writelines(my_data_list)

def read_data(month):
    folder_path = os.path.join(app.root_path, 'Hike')
    filename = os.path.join(folder_path, f"{month}.csv")
    data = {}
    with open(filename, 'r') as file:
        lines = file.readlines()[1:]  # Skip header
        for line in lines:
            emp_id, role, quantity, time_worked = line.strip().split(',')
            data[(emp_id, role)] = (quantity, time_worked)
    return data


def play_alert_sound():
    mixer.music.load("music.wav")

    while True:
        if not mixer_queue.empty():
            mixer.music.play()
            mixer_queue.get()
        time.sleep(1)

mixer_thread = threading.Thread(target=play_alert_sound)
mixer_thread.daemon = True
mixer_thread.start()

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.20
frame_check = 100
message_check = 200
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

flag = 0
guard_awake = True


def send_whatsapp_message(message):
    recipient_number = "+918856025205"
    try:
        kit.sendwhatmsg_instantly(recipient_number, message, tab_close=True)
        print("Message sent successfully")
    except Exception as e:
        print(f"Error: {str(e)}")

def identify_guard(frame):
    global guard_id

    # Find face locations and encodings in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Iterate over each face found in the frame
    for face_encoding in face_encodings:
        # Compare face encoding with known encodings
        matches = face_recognition.compare_faces(encodedFaceKnown, face_encoding)

        # Check if there's a match
        if True in matches:
            match_index = matches.index(True)
            guard_id = employee_ids[match_index]
            return guard_id  # Return the ID of the recognized guard


def update_penalty(guard_id):
    # Get the current date
    current_date = dt_module.datetime.now().strftime('%d-%m-%Y')

    # Get the current penalty count for the guard from the Firebase database
    current_penalty_ref = db.reference(f'Surveillance/{guard_id}/penalty/total_penalty')
    current_penalty = current_penalty_ref.get()

    if current_penalty is None:
        # If the current penalty count is None, set it to 0
        current_penalty = 0

    # Increment the total penalty count by 1
    updated_total_penalty = current_penalty + 1

    # Update the total penalty count in the Firebase database
    current_penalty_ref.set(updated_total_penalty)

    # Increment the penalty count for the current date
    date_ref = db.reference(f'Surveillance/{guard_id}/penalty/date/{current_date}')
    current_date_penalty = date_ref.get()

    if current_date_penalty is None:
        # If the penalty count for the current date is None, set it to 0
        current_date_penalty = 0

    # Increment the penalty count for the current date by 1
    date_ref.set(current_date_penalty + 1)


def detect_drowsiness(frame):
    global flag, guard_awake

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if guard_awake:
                    mixer_queue.put(True)
        else:
            flag = 0

        if flag >= message_check:
            guard_id_detected = identify_guard(frame)
            if guard_id_detected is not None:  # Check if guard is detected
                update_penalty(guard_id_detected)
                send_whatsapp_message(f"Heads up! Security guard {guard_id_detected} is currently sleeping/drowsy. "
                                      f"Can someone from security do a quick check?")  # Send message with guard ID
            else:
                send_whatsapp_message("Heads up! Some Security guard is currently sleeping/drowsy. Can someone from "
                                      "security do a quick check?")  # Send generic message
            flag = 0

    return frame

def generate_video_frames():
    cap = camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        frame = detect_drowsiness(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/attendance_data')
def attendance_data():
    dt_string = time.strftime('%d-%B-%Y')
    filename = os.path.join("Attendance", f"{dt_string}.csv")

    attendance_data = []

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip the header line
                columns = line.strip().split(',')
                attendance_data.append({
                    'id': columns[0],
                    'Name': columns[1],
                    'Time': dt_class.strptime(columns[2], "%H:%M:%S").strftime("%I:%M:%S %p"),
                    'Total Time': humanize.precisedelta(int(columns[3]), minimum_unit="seconds")
                })

        return jsonify({'attendance_data': attendance_data})
    else:
        return jsonify({'message': "No attendance data available for today."})

def get_guard_attendance_data():
    # Get today's date
    today_date = dt_class.now().strftime('%d-%m-%Y')

    # Fetch data from Firebase
    surveillance_ref = db.reference('Surveillance')
    surveillance_data = surveillance_ref.get()

    # Process the data to extract guard IDs, names, and penalties for the current day
    attendance_data = []
    for guard_id, guard_data in surveillance_data.items():
        penalties = guard_data.get('penalty', {}).get('date', {}).get(today_date, 0)
        attendance_data.append({
            'id': guard_id,
            'name': guard_data.get('name', ''),
            'penalties': penalties
        })

    # Return the data as JSON
    return jsonify({'attendance_data': attendance_data})

@app.route('/guard_attendance_data')
def guard_attendance_data():
    return get_guard_attendance_data()

@app.route('/')
def index():
    ref = db.reference('Employee Attendance')  # Reference to 'Employee Attendance' node
    employee_data = ref.get()  # Fetch employee details from Firebase Realtime Database

    if employee_data:
        for employee_id, employee_details in employee_data.items():
            bucket = storage.bucket()
            blob = bucket.get_blob(f'Images/{employee_id}.jpg')
            if blob:
                expiration = dt_module.timedelta(hours=1)
                signed_url = blob.generate_signed_url(expiration=expiration)
                employee_details['image_url'] = signed_url
            else:
                print(f"Image not found for {employee_id}")

    return render_template('index.html', employee_data=employee_data if 'employee_data' in locals() else None,  is_admin_logged_in=is_admin_logged_in)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/ContactUs')
def contact():
    return render_template('ContactUs.html')

@app.route('/AboutUs')
def about():
    return render_template('AboutUs.html')



@app.route('/user_info/<employee_id>')
def info(employee_id):
    global is_admin_logged_in

    if not is_admin_logged_in:
        return redirect(url_for('login'))  # Redirect to login page if not logged in as admin

    ref = db.reference('Employee Attendance')
    bucket = storage.bucket()
    blob = bucket.get_blob(f'Images/{employee_id}.jpg')
    employee_details = ref.child(employee_id).get()
    if blob:
        expiration = dt_module.timedelta(hours=1)
        signed_url = blob.generate_signed_url(expiration=expiration)
        employee_details['image_url'] = signed_url
    return render_template('employee_info.html', user_details=employee_details)


@app.route('/submit-hike', methods=['POST'])
def submit_hike():
    global is_admin_logged_in

    if not is_admin_logged_in:
        return redirect(url_for('login'))  # Redirect to login page if not logged in as admin

    if request.method == 'POST':
        month = request.form['month']
        employee_id = request.form['employee_id']
        role = request.form['role']

        # Read data from file for the given month
        data = read_data(month)

        # Check if employee ID and role match
        key = (employee_id, role)
        if key in data:
            quantity_produced, time_worked = data[key]
            with open('preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
            new_data = pd.DataFrame({
                'Quantity_Produced': [quantity_produced],
                'Time_Worked': [time_worked],
                'Role': [role]
            })
            processed_new_data = preprocessor.transform(new_data)
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            predictions = model.predict(processed_new_data)
            success_message =  f"Hike should be: {predictions}%"
            return render_template('services.html', success_message=success_message)
        else:
            error_message = "Employee ID and Role not found in the data."
            return render_template('services.html', error_message=error_message)


@app.route('/surveillance')
def index_sur():
    return render_template('index_sur.html')


@app.route('/video_feed_sur')
def video_feed_sur():
    return Response(generate_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_buzzer', methods=['POST'])
def stop_buzzer():
    while not mixer_queue.empty():
        mixer_queue.get()  # Clear the queue
    return "Buzzer stopped."


if __name__ == "__main__":
    print("Running on http://localhost:5000/")
    app.run(host='0.0.0.0', debug=True, use_reloader=False)
