import os
from base64 import b64encode, b64decode
from Crypto.Cipher import AES, DES
from Crypto.Util.Padding import pad
from Crypto.Util.Padding import unpad
import hashlib


class CryptoBase:
    def __init__(self) -> None:
        pass
    
    def encrypt_str(self, src_str:bytes)->bytes:
        # 返回字符串加密后的字符串(base64)
        encrypt_str = b''
        return encrypt_str

    def encrypt_file2str(self, src_file):
        if not os.path.exists(src_file):
            return None
        with open(src_file, 'rb') as f:
            filedata = f.read()
        encrypt_data = self.encrypt_str(filedata)
        return encrypt_data
    
    def encrypt_file2file(self, src_file, encrypt_file):
        encrypt_data = self.encrypt_file2str(src_file)
        with open(encrypt_file, 'wb') as f:
            f.write(encrypt_data)

    def decrypt_str(self, encrypt_str:bytes)->bytes:
        # 返回解密后的字符串
        decrypt_str = b''
        return decrypt_str

    def decrypt_file2str(self, encrypt_file):
        if not os.path.exists(encrypt_file):
            return None
        with open(encrypt_file, 'rb') as f:
            filedata = f.read()
        decrypt_data = self.decrypt_str(filedata)
        return decrypt_data
    
    def decrypt_file2file(self, encrypt_file, decrypt_file):
        decrypt_data = self.decrypt_file2str(encrypt_file)
        with open(decrypt_file, 'wb') as f:
            f.write(decrypt_data)

class AesCrypto(CryptoBase):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def new_aes(self, key, iv):
        aes = AES.new(key, AES.MODE_CBC, iv)
        return aes

    def encrypt_str(self, src_str:bytes):
        # 返回字符串加密后的字符串(base64)
        hash_key = hashlib.md5(src_str).hexdigest().encode('utf-8')
        aes = self.new_aes(hash_key[:16], hash_key[16:])
        padtext = pad(src_str, 16, style='pkcs7')
        encrypt_str = aes.encrypt(padtext)
        encrypt_str = b64encode(hash_key+encrypt_str)
        return encrypt_str
    
    def decrypt_str(self, encrypt_str:bytes):
        """解密"""
        encrypt_str = b64decode(encrypt_str)
        hash_key = encrypt_str[:32]
        encrypt_str = encrypt_str[32:]
        aes = self.new_aes(hash_key[:16], hash_key[16:])
        padtext = aes.decrypt(encrypt_str)
        decrypt_str = unpad(padtext, 16, 'pkcs7')
        return decrypt_str

def main():
    ## aes
    aes = AesCrypto()
    encrypt_data = aes.encrypt_str("test".encode("utf-8"))
    print(encrypt_data)
    decrypt_str = aes.decrypt_str(encrypt_data)
    print(decrypt_str)
    # encrypt_str = aes.encrypt_file2str("/root/sync2/data/yolov4-tiny-3l.cfg")
    # print(encrypt_str)
    # aes.encrypt_file2file(r"C:\Users\mqr\Desktop\DetectionInfer\best.onnx", 
    #                       r"C:\Users\mqr\Desktop\DetectionInfer\best.onnx.encrypt")
    # decrypt_str = aes.decrypt_file2str("/root/sync2/data/yolov4-tiny-3l.crypto")
    # print(decrypt_str)
    aes.decrypt_file2file(r"C:\Users\mqr\Desktop\DetectionInfer\TestInfer\Models\best.onnx.encrypt",
                           r"C:\Users\mqr\Desktop\DetectionInfer\TestInfer\Models\best.onnx")


if __name__ == '__main__':
    main()
    

    