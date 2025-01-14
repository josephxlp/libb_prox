import time
import psutil
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from plyer import notification

class Timer:
  def __init__(self, description=""):
      """
      Initialize the Timer with an optional description.

      :param description: A text description of the timing session.
      """
      self.description = description
      self.start_time = None
      self.end_time = None
      self.start_cpu = None
      self.start_memory = None

  def time_start(self):
      """Start the timer and record initial CPU and memory usage."""
      self.start_time = time.time()
      self.start_cpu = psutil.cpu_percent(interval=None)
      self.start_memory = psutil.virtual_memory().used
      formatted_start_time = datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')
      print("###############################")
      print(f"Time started at: {formatted_start_time}")
      if self.description:
          print(f"Description: {self.description}")

  def time_end(self):
      """Stop the timer, calculate elapsed time, and print the results."""
      if self.start_time is None:
          raise Exception("Timer has not been started.")
      self.end_time = time.time()
      elapsed_time = self.end_time - self.start_time
      end_cpu = psutil.cpu_percent(interval=None)
      end_memory = psutil.virtual_memory().used

      cpu_usage = end_cpu - self.start_cpu
      memory_usage = end_memory - self.start_memory

      formatted_start_time = datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')
      formatted_end_time = datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')

      results = (
          "###############################\n"
          f"Time started at: {formatted_start_time}\n"
          f"Time stopped at: {formatted_end_time}\n"
          f"Elapsed time: {elapsed_time:.2f} seconds\n"
          f"Elapsed time: {elapsed_time / 3600:.2f} hours\n"
          f"Elapsed time: {elapsed_time / 86400:.2f} days\n"
          f"Elapsed time: {elapsed_time / 604800:.2f} weeks\n"
          f"CPU usage change: {cpu_usage:.2f}%\n"
          f"Memory usage change: {memory_usage / (1024 ** 2):.2f} MB\n"
      )
      print(results)
      self.send_email_notification(results)
      self.send_desktop_notification(results)

  def reset(self):
      """Reset the timer and clear all recorded data."""
      self.start_time = None
      self.end_time = None
      self.start_cpu = None
      self.start_memory = None
      print("Timer reset.")

  def send_email_notification(self, message):
      """
      Send an email notification with the timing results.

      :param message: The message to be sent in the email.
      """
      # Configure your email settings
      sender_email = "your_email@example.com"
      receiver_email = "receiver_email@example.com"
      password = "your_password"

      msg = MIMEMultipart()
      msg['From'] = sender_email
      msg['To'] = receiver_email
      msg['Subject'] = "Timer Notification"

      msg.attach(MIMEText(message, 'plain'))

      try:
          with smtplib.SMTP('smtp.example.com', 587) as server:
              server.starttls()
              server.login(sender_email, password)
              server.send_message(msg)
          print("Email sent successfully.")
      except Exception as e:
          print(f"Failed to send email: {e}")

  def send_desktop_notification(self, message):
      """
      Send a desktop notification with the timing results.

      :param message: The message to be displayed in the notification.
      """
      try:
          notification.notify(
              title="Timer Notification\nScript Finished Running",
              message=message,
              app_name="Timer",
              timeout=10
          )
          print("Desktop notification sent.")
      except Exception as e:
          print(f"Failed to send desktop notification: {e}")

# Example usage:
# timer = Timer(description="Sample timing session")
# timer.time_start()
# time.sleep(2)  # Simulating a code block with a 2-second sleep
# timer.time_end()

# after email open notification on the desktop

# improve this function 
# well documented output, acccept a text also 
# send an email notification 
# make into a package and install globally here 


