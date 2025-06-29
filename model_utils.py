# model_util.py

import os, cv2, time, torch, yagmail, pygame
from datetime import datetime
from torchvision import transforms
from model import load_model
import pyttsx3, yagmail
from PIL import Image
from torchvision import transforms
#for face encodings
import pickle, numpy as np
import torchvision.models as models
import torch.nn as nn
from video_stream import VideoStream  # Import the threaded class
import time
from imutils.video import VideoStream


#fr images to email
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
import ssl



AUTHORIZED_DIR = "faces/authorised"
STRANGER_DIR = "faces/strangers"
ALERT_SOUND_PATH = "alert.mp3"  # Must exist
EMAIL = "youremail@gmail.com"
EMAIL_PASS = "your_app_password"

os.makedirs(AUTHORIZED_DIR, exist_ok=True)
os.makedirs(STRANGER_DIR, exist_ok=True)



#face encodings saved
ENCODINGS_DIR = "encodings"
os.makedirs(ENCODINGS_DIR, exist_ok=True)

#  Feature‑extractor (AlexNet without the last FC layer) – load ONCE
_feature_model = models.alexnet(pretrained=True)
_feature_model.classifier = nn.Sequential(*list(_feature_model.classifier.children())[:-1])
_feature_model.eval()

# single, shared transform for embeddings
_embed_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



model = load_model()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 110)
engine.setProperty('volume', 1.0)
def speak(message):
    print(message)
    try:
        engine.say(message)
        engine.runAndWait()
    except:
        print("Voice error")



def user_exists(name):
    return os.path.exists(os.path.join(AUTHORIZED_DIR, f"{name}.jpg"))


# def send_email_alert():
#     try:
#         yag = yagmail.SMTP(user='agrawalshubhangi03@gmail.com', password='hszx ltsu haeu jhhv')
#         yag.send(
#             to='agrawalshubhangi03@gmail.com',
#             subject='Home Security Alert',
#             contents='Stranger detected by your home security system after 5 failed attempts.'
#         )
#         speak("Alert email sent successfully!")
#     except Exception as e:
#         speak("Failed to send email alert.")
#         print("Email error:", e)

#fr images to email
def send_stranger_images(image_paths):
    try:
        yag = yagmail.SMTP(user='agrawalshubhangi03@gmail.com', password='hszx ltsu haeu jhhv')
        contents = ['5 stranger attempts detected. Images attached below.'] + image_paths
        yag.send(
            to='agrawalshubhangi03@gmail.com',
            subject='Home Security Alert - 5 Stranger Attempts',
            contents=contents
        )
        speak("Stranger images email sent successfully!")
    except Exception as e:
        speak("Failed to send stranger images email.")
        print("Stranger image email error:", e)







#previous code- face encodings saved, camera not open on authnetication, registration-no voicw
#
# def register_user(name):
#     import os
#     import cv2
#
#     if user_exists(name):
#         msg = f"User '{name}' already exists. Please choose a different name."
#         speak(msg)
#         return False, msg
#
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         msg = "Unable to access the camera."
#         speak(msg)
#         return False, msg
#
#     ret, frame = cap.read()
#     if not ret or frame is None:
#         cap.release()
#         msg = "No image captured. Please try again."
#         speak(msg)
#         return False, msg
#
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 5)
#
#     if len(faces) == 0:
#         cap.release()
#         msg = "No face detected. Please adjust yourself in front of the camera."
#         speak(msg)
#         return False, msg
#     elif len(faces) > 1:
#         faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
#         info = "Multiple faces detected. Using the largest detected face."
#         speak(info)
#     else:
#         info = None
#
#     x, y, w, h = faces[0]
#     face = frame[y:y + h, x:x + w]
#     save_path = os.path.join(AUTHORIZED_DIR, f"{name}.jpg")
#
#     try:
#         os.makedirs(AUTHORIZED_DIR, exist_ok=True)
#         cv2.imwrite(save_path, face)
#
#
#         #new block
#         # ---- save face embedding ----
#         from PIL import Image
#         face_pil = Image.fromarray(face).convert('RGB')
#         face_tensor = _embed_tf(face_pil).unsqueeze(0)
#         with torch.no_grad():
#             embedding = _feature_model(face_tensor).squeeze(0).numpy()
#
#         embed_path = os.path.join(ENCODINGS_DIR, f"{name}.npy")
#         np.save(embed_path, embedding)  # ---> encodings/<name>.npy
#
#
#
#     except Exception as e:
#         cap.release()
#         msg = f"Error saving image: {str(e)}"
#         speak(msg)
#         return False, msg
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     msg = f"{info + ' ' if info else ''}User '{name}' registered successfully."
#     speak(msg)
#     return True, msg
#
#
# def authenticate_user():
#     import os
#     import cv2
#     import time
#     import torch
#     import yagmail
#     import numpy as np
#     from PIL import Image
#     from torchvision import transforms
#
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
#
#     def send_email_alert():
#         try:
#             yag = yagmail.SMTP(user='agrawalshubhangi03@gmail.com', password='hszx ltsu haeu jhhv')
#             yag.send(
#                 to='agrawalshubhangi03@gmail.com',
#                 subject='Home Security Alert',
#                 contents='Stranger detected by your home security system after 5 failed attempts.'
#             )
#             speak("Alert email sent successfully!")
#         except Exception as e:
#             speak("Failed to send email alert.")
#             print("Email error:", e)
#
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         speak("Cannot access camera for authentication.")
#         return False, "Camera error."
#
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     frame_count = 0
#     stranger_count = 0
#     cooldown = False
#     cooldown_start = None
#
#     while True:
#         ret, frame = cap.read()
#         frame_count += 1
#         if not ret or frame_count % 5 != 0:
#             continue
#
#         if cooldown:
#             elapsed = time.time() - cooldown_start
#             if elapsed < 10:
#                 speak(f"Cooldown active. Please wait {int(10 - elapsed)} seconds.")
#                 continue
#             cooldown = False
#             stranger_count = 0
#
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#
#         if len(faces) == 0:
#             speak("No face detected. Please adjust yourself in front of the camera.")
#             continue
#
#         if len(faces) > 1:
#             speak("Multiple faces detected. Using the largest detected face.")
#             faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
#
#         for (x, y, w, h) in faces[:1]:
#             face_img = frame[y:y + h, x:x + w]
#             face_pil = Image.fromarray(cv2.resize(face_img, (224, 224))).convert('RGB')
#             face_tensor = transform(face_pil).unsqueeze(0)
#
#             # with torch.no_grad():
#             #     pred = model(face_tensor).item()
#             #
#             # print(f"Prediction Score: {pred:.4f}")
#             #
#             # if pred >= 0.7:
#             #     label = 'Authorized'
#             #     color = (0, 255, 0)
#             #     stranger_count = 0
#             #     speak("Authentication granted. Welcome back!")
#             #     cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             #     cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
#             #     cv2.imshow("Security Feed", frame)
#             #     cv2.waitKey(3000)  # Hold frame for 3 sec
#             #     cap.release()
#             #     cv2.destroyAllWindows()
#             #     return True, "Authentication granted."
#             #
#             # else:
#             #     label = 'Stranger'
#             #     color = (0, 0, 255)
#             #     stranger_count += 1
#             #     speak(f"Authentication not granted. Attempt {stranger_count} of 5.") previous one
#
#             #new block- to match encodings
#
#             # ---------- compute live embedding ----------
#             face_pil = Image.fromarray(cv2.resize(face_img, (224, 224))).convert('RGB')
#             live_tensor = _embed_tf(face_pil).unsqueeze(0)
#             with torch.no_grad():
#                 live_emb = _feature_model(live_tensor).squeeze(0).numpy()
#
#             # ---------- compare with stored encodings ----------
#             best_match, best_name, best_sim = False, None, 0.0
#             for file in os.listdir(ENCODINGS_DIR):
#                 registered_emb = np.load(os.path.join(ENCODINGS_DIR, file))
#                 sim = np.dot(live_emb, registered_emb) / (np.linalg.norm(live_emb) * np.linalg.norm(registered_emb))
#                 if sim > best_sim:
#                     best_sim, best_name = sim, file.split('.')[0]
#                     best_match = sim > 0.80  # <-- threshold, tweak if needed
#
#             if best_match:
#                 label, color = f"Authorized: {best_name}", (0, 255, 0)
#                 stranger_count = 0
#                 speak(f"Authentication granted. Welcome {best_name}.")
#             else:
#                 label, color = "Stranger", (0, 0, 255)
#                 stranger_count += 1
#                 speak(f"Authentication not granted. Attempt {stranger_count} of 5.")
#                 # (keep your snapshot / email / cooldown logic unchanged below)
#
#                 os.makedirs("strangers", exist_ok=True)
#                 filename = f"strangers/stranger_{int(time.time())}.jpg"
#                 cv2.imwrite(filename, face_img)
#                 speak("Stranger image saved.")
#
#                 if stranger_count >= 5:
#                     speak("Maximum unauthorized attempts reached.")
#                     send_email_alert()
#                     cooldown = True
#                     cooldown_start = time.time()
#
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
#
#         cv2.imshow("Security Feed", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#     return False, "Authentication not granted."

#new code



def register_user(name):  #works well, all voice, and face encodings saved
    import os
    import cv2
    import numpy as np
    from PIL import Image
    import torch

    if user_exists(name):
        msg = f"User '{name}' already exists. Please choose a different name."
        try: speak(msg)
        except: pass
        return False, msg

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        msg = "Unable to access the camera."
        try: speak(msg)
        except: pass
        return False, msg

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        msg = "No image captured. Please try again."
        try: speak(msg)
        except: pass
        return False, msg

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        msg = "No face detected. Please adjust yourself in front of the camera."
        try: speak(msg)
        except: pass
        return False, msg

    # Use largest face if multiple detected
    if len(faces) > 1:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        info = "Multiple faces detected. Using the largest detected face."
        try: speak(info)
        except: pass
    else:
        info = None

    x, y, w, h = faces[0]
    face = frame[y:y + h, x:x + w]

    try:
        os.makedirs(AUTHORIZED_DIR, exist_ok=True)
        os.makedirs(ENCODINGS_DIR, exist_ok=True)

        save_path = os.path.join(AUTHORIZED_DIR, f"{name}.jpg")
        cv2.imwrite(save_path, face)

        # Generate and save face embedding
        face_pil = Image.fromarray(cv2.resize(face, (224, 224))).convert('RGB')
        face_tensor = _embed_tf(face_pil).unsqueeze(0)

        with torch.no_grad():
            embedding = _feature_model(face_tensor).squeeze(0).numpy()

        embed_path = os.path.join(ENCODINGS_DIR, f"{name}.npy")
        np.save(embed_path, embedding)

    except Exception as e:
        msg = f"Error saving data: {str(e)}"
        try: speak(msg)
        except: pass
        return False, msg

    msg = f"{info + ' ' if info else ''}User '{name}' registered successfully."
    try: speak(msg)
    except: pass
    return True, msg

# def get_latest_frame(cap):
#     while cap.grab():
#         pass
#     ret, frame = cap.retrieve()
#     return ret, frame




# def authenticate_user():
#     stream = VideoStream()  # Start threaded capture
#     vs = VideoStream(src=0).start()
#     time.sleep(1.0)  # Let it warm up
#
#     # ✅ Model & encodings setup
#     _feature_model.eval()
#     known_encodings, known_names = [], []
#     for file in os.listdir(ENCODINGS_DIR):
#         if file.endswith(".npy"):
#             name = file.replace(".npy", "")
#             emb = np.load(os.path.join(ENCODINGS_DIR, file))
#             known_encodings.append(emb)
#             known_names.append(name)
#
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     stranger_count, cooldown, cooldown_start = 0, False, None
#
#     # while True:
#     #     ret, frame = stream.read()
#     #     if not ret:
#     #         speak("Camera read failed.")
#     #         continue
#
#     frame_count = 0
#     while True:
#         ret, frame = vs.read()
#         if not ret:
#             continue
#
#         frame_count += 1
#         if frame_count % 2 != 0:
#             continue  # Only process every second frame
#
#         if cooldown:
#             elapsed = time.time() - cooldown_start
#             if elapsed < 10:
#                 speak(f"Cooldown active. Wait {int(10 - elapsed)} seconds.")
#                 continue
#             cooldown = False
#             stranger_count = 0
#
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
#
#         if len(faces) == 0:
#             speak("No face detected.")
#             continue
#         if len(faces) > 1:
#             speak("Multiple faces detected.")
#             faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
#
#         for (x, y, w, h) in faces[:1]:
#             face_img = frame[y:y + h, x:x + w]
#             face_resized = cv2.resize(face_img, (224, 224))
#             face_tensor = _embed_tf(Image.fromarray(face_resized).convert('RGB')).unsqueeze(0)
#
#             with torch.no_grad():
#                 live_emb = _feature_model(face_tensor).squeeze(0).numpy()
#
#             similarities = [
#                 np.dot(live_emb, e) / (np.linalg.norm(live_emb) * np.linalg.norm(e))
#                 for e in known_encodings
#             ]
#
#             if similarities:
#                 max_sim = max(similarities)
#                 best_name = known_names[np.argmax(similarities)]
#
#                 if max_sim > 0.65:
#                     label, color = f"Authorized: {best_name}", (0, 255, 0)
#                     stranger_count = 0
#                     speak(f"Welcome {best_name}")
#                 else:
#                     label, color = "Stranger", (0, 0, 255)
#                     stranger_count += 1
#                     speak(f"Unauthorized attempt {stranger_count} of 5.")
#                     os.makedirs("faces/strangers", exist_ok=True)
#                     cv2.imwrite(f"faces/strangers/stranger_{int(time.time())}.jpg", face_img)
#
#                     if stranger_count >= 5:
#                         speak("Too many failed attempts. Sending alert.")
#                         send_email_alert()
#                         cooldown = True
#                         cooldown_start = time.time()
#             else:
#                 label, color = "No registered users", (255, 255, 0)
#                 speak("Please register first.")
#
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#
#         cv2.imshow("Authentication", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     stream.stop()
#     cv2.destroyAllWindows()
#     return False, "Authentication ended."


from imutils.video import VideoStream
import time
import cv2
import os
import numpy as np
from PIL import Image
import torch

# def authenticate_user(): #emial without images, just enbale send email alert nd not stranger image
#     # ✅ Start threaded video stream
#     vs = VideoStream(src=0).start()
#     time.sleep(1.0)  # Warm-up time
#
#     # ✅ Prepare model and encodings
#     _feature_model.eval()
#     known_encodings, known_names = [], []
#
#     for file in os.listdir(ENCODINGS_DIR):
#         if file.endswith(".npy"):
#             name = file.replace(".npy", "")
#             emb = np.load(os.path.join(ENCODINGS_DIR, file))
#             known_encodings.append(emb)
#             known_names.append(name)
#
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     stranger_count = 0
#     stranger_images = []
#     cooldown = False
#     cooldown_start = None
#
#     frame_count = 0
#
#     while True:
#         frame = vs.read()  # ✅ Only frame (no ret)
#
#         if frame is None:
#             continue
#
#         frame_count += 1
#         if frame_count % 2 != 0:
#             continue  # Skip every other frame to reduce load
#
#         if cooldown:
#             elapsed = time.time() - cooldown_start
#             if elapsed < 10:
#                 speak(f"Cooldown active. Wait {int(10 - elapsed)} seconds.")
#                 continue
#             cooldown = False
#             stranger_count = 0
#
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
#
#         if len(faces) == 0:
#             speak("No face detected.")
#             continue
#         if len(faces) > 1:
#             speak("Multiple faces detected.")
#             faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
#
#         for (x, y, w, h) in faces[:1]:
#             face_img = frame[y:y + h, x:x + w]
#             face_resized = cv2.resize(face_img, (224, 224))
#             face_tensor = _embed_tf(Image.fromarray(face_resized).convert('RGB')).unsqueeze(0)
#
#             with torch.no_grad():
#                 live_emb = _feature_model(face_tensor).squeeze(0).numpy()
#
#             similarities = [
#                 np.dot(live_emb, e) / (np.linalg.norm(live_emb) * np.linalg.norm(e))
#                 for e in known_encodings
#             ]
#
#             if similarities:
#                 max_sim = max(similarities)
#                 best_name = known_names[np.argmax(similarities)]
#
#                 if max_sim > 0.65:
#                     label, color = f"Authorized: {best_name}", (0, 255, 0)
#                     stranger_count = 0
#                     speak(f"Welcome {best_name}")
#                 else:
#                     label, color = "Stranger", (0, 0, 255)
#                     stranger_count += 1
#                     speak(f"Unauthorized attempt {stranger_count} of 5.")
#                     os.makedirs("faces/strangers", exist_ok=True)
#                     cv2.imwrite(f"faces/strangers/stranger_{int(time.time())}.jpg", face_img)
#
#                     if stranger_count >= 5:
#                         speak("Too many failed attempts. Sending alert.")
#                         # send_email_alert()
#                         cooldown = True
#                         cooldown_start = time.time()
#             else:
#                 label, color = "No registered users", (255, 255, 0)
#                 speak("Please register first.")
#
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#
#         cv2.imshow("Authentication", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     vs.stop()
#     cv2.destroyAllWindows()
#     return False, "Authentication ended."



def authenticate_user():
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    _feature_model.eval()
    known_encodings, known_names = [], []

    for file in os.listdir(ENCODINGS_DIR):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            emb = np.load(os.path.join(ENCODINGS_DIR, file))
            known_encodings.append(emb)
            known_names.append(name)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frame_count = 0
    stranger_count = 0
    stranger_images = []
    cooldown = False
    cooldown_start = None

    while True:
        frame = vs.read()
        if frame is None:
            continue

        frame_count += 1
        if frame_count % 2 != 0:
            continue

        if cooldown:
            elapsed = time.time() - cooldown_start
            if elapsed < 10:
                speak(f"Cooldown active. Wait {int(10 - elapsed)} seconds.")
                continue
            cooldown = False
            stranger_count = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        if len(faces) == 0:
            speak("No face detected.")
            continue
        if len(faces) > 1:
            speak("Multiple faces detected.")
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

        for (x, y, w, h) in faces[:1]:
            face_img = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face_img, (224, 224))
            face_tensor = _embed_tf(Image.fromarray(face_resized).convert('RGB')).unsqueeze(0)

            with torch.no_grad():
                live_emb = _feature_model(face_tensor).squeeze(0).numpy()

            similarities = [
                np.dot(live_emb, e) / (np.linalg.norm(live_emb) * np.linalg.norm(e))
                for e in known_encodings
            ]

            if similarities:
                max_sim = max(similarities)
                best_name = known_names[np.argmax(similarities)]

                if max_sim > 0.65:
                    label, color = f"Authorized: {best_name}", (0, 255, 0)
                    stranger_count = 0
                    stranger_images.clear()
                    speak(f"Welcome {best_name}")
                else:
                    label, color = "Stranger", (0, 0, 255)
                    stranger_count += 1
                    speak(f"Unauthorized attempt {stranger_count} of 5.")
                    os.makedirs("faces/strangers", exist_ok=True)

                    timestamp = int(time.time())
                    img_path = f"faces/strangers/stranger_{timestamp}.jpg"
                    cv2.imwrite(img_path, face_img)

                    stranger_images.append(img_path)
                    if len(stranger_images) > 5:
                        stranger_images.pop(0)

                    if stranger_count >= 5:
                        speak("Too many failed attempts. Sending alert with images.")
                        send_stranger_images(stranger_images)  # ⬅️ NEW MULTI IMAGE FUNCTION
                        cooldown = True
                        cooldown_start = time.time()
                        stranger_count = 0
                        stranger_images.clear()
            else:
                label, color = "No registered users", (255, 255, 0)
                speak("Please register first.")

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Authentication", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()
    return False, "Authentication ended."
