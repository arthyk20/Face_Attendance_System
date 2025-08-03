from keras_facenet import FaceNet
import cv2
import numpy as np
import pickle
from datetime import datetime, timedelta
import pandas as pd
from email_utils import send_email
from playsound import playsound
import os

# === Setup ===
embedder = FaceNet()
encoder, classifier = pickle.load(open("D:/face_detection_system/embeddings/face_classifier.pkl", "rb" ))
student_df = pd.read_csv("D:/face_detection_system/student.csv")
expected_people = dict(zip(student_df['name'], student_df['email']))

attendance = {}      # {'name': {'in': ..., 'out': ...}}
last_seen = {}       # {'name': datetime}
emailed = set()      # track who’s been emailed

# Time thresholds
cutoff_time = datetime.strptime("09:00:00", "%H:%M:%S").time()
grace_time  = (datetime.combine(datetime.today(), cutoff_time) + timedelta(minutes=20)).time()  # 09:20
absent_time = (datetime.combine(datetime.today(), cutoff_time) + timedelta(minutes=30)).time()  # 09:30

start_time = datetime.now()

# === Ensure directory exists ===
save_dir = r"D:/face_detection_system"
os.makedirs(save_dir, exist_ok=True)

# === CSV file setup ===
csv_path = os.path.join(save_dir, "attendance_logs.csv")
if not os.path.exists(csv_path):
    pd.DataFrame(columns=["Name", "In", "Out", "Date", "Status"]) \
      .to_csv(csv_path, index=False)

# === Start Webcam ===
cap = cv2.VideoCapture(0)
print("🎥 Starting face recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = datetime.now()
    frame_time = now
    results = embedder.extract(frame)
    seen_this_frame = set()

    # --- Recognition & IN logic ---
    if results:
        for r in results:
            emb  = r['embedding']
            preds = classifier.predict_proba([emb])[0]
            j    = np.argmax(preds)
            prob = preds[j]

            if prob > 0.50:
                name = encoder.inverse_transform([j])[0]
                seen_this_frame.add(name)

                # IN timestamp
                if name not in attendance:
                    in_time_str = frame_time.strftime("%H:%M:%S")
                    attendance[name] = {'in': in_time_str, 'out': None}

                    # play beep
                    try:
                        playsound(os.path.join(save_dir, 'beep.wav'))
                    except:
                        print("⚠️ Beep failed (check beep.wav)")

                    # Determine status
                    in_t = datetime.strptime(in_time_str, "%H:%M:%S").time()
                    status = "Late" if in_t > grace_time else "Present"

                    # Append IN to CSV
                    df = pd.read_csv(csv_path)
                    today = frame_time.strftime("%Y-%m-%d")
                    mask  = (df["Name"] == name) & (df["Date"] == today)
                    if df[mask].empty:
                        new_row = pd.DataFrame([{
                            "Name": name,
                            "In": in_time_str,
                            "Out": "",
                            "Date": today,
                            "Status": status
                        }])
                        new_row.to_csv(csv_path, mode='a', index=False, header=False)
                        print(f"✅ IN saved for {name} at {in_time_str} ({status})")
                    else:
                        print(f"ℹ️ {name} already checked in today.")

                # update last seen
                last_seen[name] = frame_time

                # draw box
                x, y, w, h = r['box']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, f"{name} {prob*100:.1f}%", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # --- OUT logic (30 s after last seen) ---
    for name, last_time in list(last_seen.items()):
        if name not in seen_this_frame and (frame_time - last_time) > timedelta(seconds=30):
            if attendance[name]['out'] is None:
                out_time_str = frame_time.strftime("%H:%M:%S")
                attendance[name]['out'] = out_time_str

                # update CSV
                df = pd.read_csv(csv_path)
                today = frame_time.strftime("%Y-%m-%d")
                mask  = (df["Name"] == name) & (df["Date"] == today)
                df.loc[mask, "Out"] = out_time_str
                df.to_csv(csv_path, index=False)
                print(f"✅ OUT saved for {name} at {out_time_str}")

            del last_seen[name]

    # --- Email notifications and Absent marking ---
    for name, email in expected_people.items():
        today = frame_time.strftime("%Y-%m-%d")

        # ABSENT Check (after 9:30)
        if now.time() > absent_time and name not in attendance and name not in emailed:
            # Add to CSV
            df = pd.read_csv(csv_path)
            mask = (df["Name"] == name) & (df["Date"] == today)
            if df[mask].empty:
                new_row = pd.DataFrame([{
                    "Name": name,
                    "In": "",
                    "Out": "",
                    "Date": today,
                    "Status": "Absent"
                }])
                new_row.to_csv(csv_path, mode='a', index=False, header=False)
                print(f"🚫 ABSENT marked in CSV for {name}")

            # Send absent mail
            send_email(email, "Absent Alert",
                       f"Hi {name}, you were not recognized by 9:30 AM and are marked absent.")
            print(f"📧 Absent email sent to {name}")
            emailed.add(name)

        # LATE Check
        elif (now.time() > grace_time
              and name in attendance
              and datetime.strptime(attendance[name]['in'], "%H:%M:%S").time() > grace_time
              and name not in emailed):
            send_email(email, "Late Alert",
                       f"Hi {name}, you checked in at {attendance[name]['in']} (after 9:20 AM).")
            print(f"📧 Late email sent to {name}")
            emailed.add(name)

    # display webcam frame
    cv2.imshow("Face Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
print("🛑 System closed.")
