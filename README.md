# Face Attendance System 👨‍💼📸
A real-time face recognition-based attendance system using FaceNet, OpenCV, and Python. Automatically detects and logs attendance with In/Out time, and sends email notifications to absentees and latecomers.

🔧 Features
🎥 Real-time face detection using webcam

🧠 Face recognition with FaceNet embeddings

📁 Logs In/Out time in Excel

📧 Sends:

Late arrival emails (after 15 mins)

Absentee emails (after 30 mins)

🗂 Stores known faces and recognizes them accurately

📦 Easy to use and run on local machine

# 📁 Folder Structure
Face_Attendance_System/
│
├── attendance.xlsx          # Realtime log file
├── embeddings.pkl           # Face embeddings
├── email_utils.py           # Email sending logic
├── face_recognition.py      # Core logic (FaceNet, recognition)
├── train_model.py           # Model training using labeled images
├── requirements.txt         # Python dependencies
├── known_faces/             # Folder with labeled images for each person
│   └── PersonName/
│       ├── 1.jpg
│       └── 2.jpg
└── ...
# 🚀 How to Run
1. Clone the repo
git clone https://github.com/yourusername/Face_Attendance_System.git
cd Face_Attendance_System
2. Create Virtual Environment (optional but recommended)
python -m venv faceenv
faceenv\Scripts\activate   # Windows
3. Install Dependencies
pip install -r requirements.txt
4. Add Known Faces
Create a folder known_faces/

Inside that, create subfolders for each person.

Put multiple images of the same person in their respective folder.

Example:
known_faces/
├── Arthy/
│   ├── 1.jpg
│   └── 2.jpg
├── Gokul/
│   ├── 1.jpg
│   └── 2.jpg
5. Train the Model
python train_model.py
6. Run the Attendance System
python face_recognition.py
# 📬 Email Notification Rules
If a recognized person comes 15+ minutes late, an email is sent.

If a person is not recognized within 30 minutes, an absent email is sent.

Email sending is handled via email_utils.py.

# 📝 Output
Excel File: attendance.xlsx with Name, Date, In Time, Out Time

Email Alerts to latecomers and absentees

# 💡 Tech Stack
Python 3.x

OpenCV

Keras-FaceNet

Scikit-learn

Pandas

smtplib (for sending emails)
