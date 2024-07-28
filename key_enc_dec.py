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

'''
Original message: Hello, World!
Encrypted message: b'gAAAAABgZ7...'
Decrypted message: Hello, World!

Summary
• Key Generation: Fernet.generate_key() creates a 32-byte symmetric key.
• Cipher Initialization: Fernet(KEY) creates a cipher object for encryption and decryption.
• Encrypting: CIPHER.encrypt(log.encode()) encrypts the log message using the cipher.
• Decrypting: To decrypt, the same key and cipher would be used with the decrypt method.
This ensures that the log data is securely encrypted and can only be decrypted by someone with the correct key, providing confidentiality and integrity.
'''

