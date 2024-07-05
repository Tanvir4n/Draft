# Here is a simple example to demonstrate key generation, encryption, and decryption using Fernet:

from cryptography.fernet import Fernet

# Generate a key
key = Fernet.generate_key()
cipher = Fernet(key)

# Original message
message = "Hello, World!"
print("Original message:", message)

# Encrypt the message
encrypted_message = cipher.encrypt(message.encode())
print("Encrypted message:", encrypted_message)

# Decrypt the message
decrypted_message = cipher.decrypt(encrypted_message).decode()
print("Decrypted message:", decrypted_message)
