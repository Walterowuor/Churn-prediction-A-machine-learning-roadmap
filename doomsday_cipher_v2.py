import os
import hmac
import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.argon2 import Argon2
from stegano import lsb
import secrets

# Fake header to spook attackers
DOOMSDAY_HEADER = b"DOOMSDAY PROTOCOL v13 - ABANDON HOPE"

def generate_ecc_keys():
    private_key = ec.generate_private_key(ec.SECP256K1())
    public_key = private_key.public_key()
    return private_key, public_key

def derive_key(passphrase: bytes) -> bytes:
    argon2 = Argon2(memory=256*1024, iterations=4, parallelism=2, salt=secrets.token_bytes(16))
    return argon2.derive(passphrase)

def encrypt_message(message: bytes, passphrase: str, public_key, image_path: str) -> str:
    # Derive symmetric key
    key = derive_key(passphrase.encode())
    
    # Generate nonce and fake header
    nonce = secrets.token_bytes(12)
    payload = DOOMSDAY_HEADER + nonce + message
    
    # AES-256-GCM or ChaCha20-Poly1305 (random choice for terror)
    if secrets.randbits(1):
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce))
    else:
        cipher = Cipher(algorithms.ChaCha20(key, nonce), modes.ChaCha20Poly1305())
    
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(payload) + encryptor.finalize()
    tag = encryptor.tag or b''  # ChaCha20 includes tag in ciphertext
    
    # Encrypt key with ECIES
    shared_key = ec.generate_private_key(ec.SECP256K1())
    shared_pub = shared_key.public_key()
    ecdh_key = shared_key.exchange(ec.ECDH(), public_key)
    key_cipher = Cipher(algorithms.AES(ecdh_key[:32]), modes.GCM(nonce))
    key_encryptor = key_cipher.encryptor()
    enc_key = key_encryptor.update(key) + key_encryptor.finalize()
    
    # HMAC for integrity
    h = hmac.new(key, digestmod=hashlib.sha256)
    h.update(ciphertext + enc_key + tag)
    mac = h.digest()
    
    # Embed in image with stegano
    output_image = f"doomsday_{secrets.token_hex(8)}.png"
    lsb.hide(image_path, ciphertext + enc_key + tag + mac + shared_pub.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )).save(output_image)
    
    return output_image

def decrypt_message(image_path: str, passphrase: str, private_key) -> bytes:
    # Extract data from image
    data = lsb.reveal(image_path)
    pub_key_len = 91  # secp256k1 DER public key length
    ciphertext, enc_key, tag, mac, peer_pub_bytes = (
        data[:-pub_key_len-32-16], data[-pub_key_len-32-16:-pub_key_len-32],
        data[-pub_key_len-32:-pub_key_len], data[-pub_key_len:-pub_key_len],
        data[-pub_key_len:]
    )
    
    # Verify HMAC
    key = derive_key(passphrase.encode())
    h = hmac.new(key, digestmod=hashlib.sha256)
    h.update(ciphertext + enc_key + tag)
    if not hmac.compare_digest(h.digest(), mac):
        raise ValueError("Integrity check failed. You’re not worthy, Buddy.")
    
    # Decrypt symmetric key with ECIES
    peer_pub = serialization.load_der_public_key(peer_pub_bytes)
    ecdh_key = private_key.exchange(ec.ECDH(), peer_pub)
    key_cipher = Cipher(algorithms.AES(ecdh_key[:32]), modes.GCM(nonce))
    key_decryptor = key_cipher.decryptor()
    dec_key = key_decryptor.update(enc_key) + key_decryptor.finalize()
    
    # Decrypt payload (try AES, then ChaCha20)
    nonce = ciphertext[len(DOOMSDAY_HEADER):len(DOOMSDAY_HEADER)+12]
    payload = ciphertext[len(DOOMSDAY_HEADER)+12:]
    try:
        cipher = Cipher(algorithms.AES(dec_key), modes.GCM(nonce, tag))
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(payload) + decryptor.finalize()
    except ValueError:
        cipher = Cipher(algorithms.ChaCha20(dec_key, nonce), modes.ChaCha20Poly1305())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(payload) + decryptor.finalize()
    
    return plaintext

if __name__ == "__main__":
    # Example usage
    private_key, public_key = generate_ecc_keys()
    message = b"Reality’s firewall is toast, Buddy!"
    image = encrypt_message(message, "WubbaLubbaDubDub", public_key, "input.png")
    print(f"Encrypted to {image}")
    decrypted = decrypt_message(image, "WubbaLubbaDubDub", private_key)
    print(f"Decrypted: {decrypted.decode()}")