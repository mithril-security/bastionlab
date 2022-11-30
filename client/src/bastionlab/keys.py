from typing import Optional
import os
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec, types


class PublicKey:
    __key: types.PUBLIC_KEY_TYPES
    __hash: bytes

    def __init__(self, key: types.PUBLIC_KEY_TYPES):
        self.__key = key
        hash = hashes.Hash(hashes.SHA256())
        hash.update(self.bytes)
        self.__hash = hash.finalize()

    def __eq__(self, o: object) -> bool:
        return self.__key.__eq__(o)

    def verify(self, signature: bytes, data: bytes) -> None:
        self.__key.verify(
            signature, data, signature_algorithm=ec.ECDSA(hashes.SHA256())
        )

    @property
    def hash(self) -> bytes:
        return self.__hash

    @property
    def bytes(self) -> bytes:
        return self.__key.public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )

    @staticmethod
    def from_pem(path: str) -> "PublicKey":
        with open(path, "rb") as f:
            return PublicKey.from_pem_content(f.read())

    def save_pem(self, path: str) -> "PublicKey":
        with open(path, "wb") as f:
            f.write(
                self.__key.public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )

    @property
    def pem(self) -> str:
        return str(
            self.__key.public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            ),
            "utf-8",
        )

    @staticmethod
    def from_bytes_content(content: bytes) -> "PublicKey":
        return PublicKey(serialization.load_der_public_key(content))

    @staticmethod
    def from_pem_content(content: bytes) -> "PublicKey":
        return PublicKey(serialization.load_pem_public_key(content))


class SigningKey:
    __key: types.PRIVATE_KEY_TYPES
    __pubkey: PublicKey

    def __init__(self, privkey: types.PRIVATE_KEY_TYPES):
        self.__key = privkey
        self.__pubkey = PublicKey(self.__key.public_key())

    def sign(self, data: bytes) -> bytes:
        return self.__key.sign(data, signature_algorithm=ec.ECDSA(hashes.SHA256()))

    def __eq__(self, o: object) -> bool:
        return self.__key.__eq__(o)

    @property
    def pubkey(self) -> PublicKey:
        return self.__pubkey

    @staticmethod
    def generate() -> "SigningKey":
        return SigningKey(ec.generate_private_key(ec.SECP256R1()))

    @staticmethod
    def keygen(path: str, password: Optional[bytes] = None) -> "SigningKey":
        signing_key_path = path
        pub_key_path = path + ".pub"
        if os.path.exists(signing_key_path):
            return SigningKey.from_pem(signing_key_path, password)
        else:
            priv_key = SigningKey.generate().save_pem(signing_key_path, password)
            priv_key.pubkey.save_pem(pub_key_path)
            return priv_key

    @staticmethod
    def from_pem(path: str, password: Optional[bytes] = None) -> "SigningKey":
        with open(path, "rb") as f:
            return SigningKey.from_pem_content(f.read(), password)

    def save_pem(self, path: str, password: Optional[bytes] = None) -> "SigningKey":
        with open(path, "wb") as f:
            f.write(
                self.__key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.BestAvailableEncryption(password)
                    if password
                    else serialization.NoEncryption(),
                )
            )
        return self

    @staticmethod
    def from_pem_content(
        content: bytes, password: Optional[bytes] = None
    ) -> "SigningKey":
        return SigningKey(serialization.load_pem_private_key(content, password))


class Identity:
    @staticmethod
    def create(
        name: Optional[str] = "bastionlab-identity", password: Optional[bytes] = None
    ) -> SigningKey:
        return SigningKey.keygen(name, password)

    @staticmethod
    def load(name: str) -> SigningKey:
        return SigningKey.from_pem(name, None)
