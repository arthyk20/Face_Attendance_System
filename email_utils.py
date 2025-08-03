# email_utils.py
import smtplib
from email.message import EmailMessage

def send_email(to_email, subject, body):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = 'arthyk938@gmail.com'
    msg['To'] = to_email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login('arthyk938@gmail.com', 'mfkyfyjusgnvqmng')
        smtp.send_message(msg)
