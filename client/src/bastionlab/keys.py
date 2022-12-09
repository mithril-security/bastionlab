from typing import Optional
import os
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec, types


class PublicKey:
    # A class for representing a public key. This class provides methods for
    # encrypting and verifying messages, as well as converting the key to and from
    # various formats (e.g. bytes, PEM).
    #
    # Attributes:
    #     _key (types.PUBLIC_KEY_TYPES): The underlying key type, represented using
    #         the `types.PUBLIC_KEY_TYPES` type.
    #     _hash (bytes): The hash of the key, used for identifying the key.

    _key: types.PUBLIC_KEY_TYPES
    _hash: bytes

    def __init__(self, key: types.PUBLIC_KEY_TYPES):
        # Initialize a `PublicKey` instance with a given public key type.

        # Args:
        #     key: An EC public key type.
        self._key = key
        hash = hashes.Hash(hashes.SHA256())
        hash.update(self.bytes)
        self._hash = hash.finalize()

    def __eq__(self, o: object) -> bool:
        # Compare this `PublicKey` instance with another object for equality.
        #
        # Args:
        #     o: The object to compare with.
        #
        # Returns:
        #     True if the objects are equal, False otherwise.
        return self._key.__eq__(o)

    def verify(self, signature: bytes, data: bytes) -> None:
        # Verify that the given signature is valid for the given data.

        # Args:
        #     signature: A signature to verify.
        #     data: The data that the signature should be for.

        # Raises:
        #     ValueError: if the signature is not valid for the given data.
        self._key.verify(signature, data, signature_algorithm=ec.ECDSA(hashes.SHA256()))

    @property
    def hash(self) -> bytes:
        # Get the hash of this `PublicKey` instance.

        # Returns:
        #     The hash of this `PublicKey` instance.
        return self._hash

    @property
    def bytes(self) -> bytes:
        # Get the DER encoding of this `PublicKey` instance.

        # Returns:
        #     The DER encoding of this `PublicKey` instance.
        return self._key.public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )

    @staticmethod
    def from_pem(path: str) -> "PublicKey":
        # Load a `PublicKey` instance from a PEM-encoded file.

        # Args:
        #     path: The path to the file to load the key from.

        # Returns:
        #     The `PublicKey` instance loaded from the given file.
        with open(path, "rb") as f:
            return PublicKey.from_pem_content(f.read())

    def save_pem(self, path: str) -> "PublicKey":
        # Save this `PublicKey` instance to a PEM-encoded file.

        # Args:
        #     path: The path to save the key to.

        # Returns:
        #     This `PublicKey` instance.
        with open(path, "wb") as f:
            f.write(
                self._key.public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )

    @property
    def pem(self) -> str:
        # Get the PEM encoding of this `PublicKey` instance.

        # Returns:
        #     The PEM encoding of this `PublicKey` instance.
        return str(
            self._key.public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            ),
            "utf-8",
        )

    @staticmethod
    def from_bytes_content(content: bytes) -> "PublicKey":
        # Load a `PublicKey` instance from a DER-encoded byte string.

        # Args:
        #     content: The DER-encoded byte string to load the key from.

        # Returns:
        #     The `PublicKey` instance loaded from the given byte string.
        return PublicKey(serialization.load_der_public_key(content))

    @staticmethod
    def from_pem_content(content: bytes) -> "PublicKey":
        # Load a `PublicKey` instance from a PEM-encoded byte string.

        # Args:
        #     content: The PEM-encoded byte string to load the key from.

        # Returns:
        #     The `PublicKey` instance loaded from the given byte string.
        return PublicKey(serialization.load_pem_public_key(content))


class SigningKey:
    # """
    # A class for representing a signing key. This class is used for creating digital
    # signatures and verifying them. It contains both the private key (used for signing)
    # and the corresponding public key (used for verification).

    # Attributes:
    #     _key (types.PRIVATE_KEY_TYPES): The private key type, represented using the
    #         `types.PRIVATE_KEY_TYPES` type.
    #     _pubkey (PublicKey): The corresponding public key, used for verifying signatures."""

    _key: types.PRIVATE_KEY_TYPES
    _pubkey: PublicKey

    def __init__(self, privkey: types.PRIVATE_KEY_TYPES):
        # Initialize a `SigningKey` instance with a given private key type.

        # Args:
        #     privkey: A private key type.
        self._key = privkey
        self._pubkey = PublicKey(self._key.public_key())

    def sign(self, data: bytes) -> bytes:
        # Sign the given data with this `SigningKey` instance's private key.

        # Args:
        #     data: The data to sign.

        # Returns:
        #     The signature for the given data.
        return self._key.sign(data, signature_algorithm=ec.ECDSA(hashes.SHA256()))

    def __eq__(self, o: object) -> bool:
        return self._key.__eq__(o)

    @property
    def pubkey(self) -> PublicKey:
        # Get the public key associated with this `SigningKey` instance.

        # Returns:
        #     The public key associated with this `SigningKey` instance.
        return self._pubkey

    @staticmethod
    def generate() -> "SigningKey":
        # Generate a new `SigningKey` instance.

        # Returns:
        #     A new `SigningKey` instance.
        return SigningKey(ec.generate_private_key(ec.SECP256R1()))

    @staticmethod
    def keygen(path: str, password: Optional[bytes] = None) -> "SigningKey":
        # Generate a new signing key and save it to the given file.

        # Args:
        #     path: The path to the file to save the signing key to.
        #     password: The password to use to encrypt the signing key. If not provided,
        #         the key will not be encrypted.

        # Returns:
        #     The generated signing key.
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
        # Load a `PublicKey` instance from a PEM-encoded file.

        # Args:
        #     path: The path to the file to load the key from.

        # Returns:
        #     The `PublicKey` instance loaded from the given file.
        with open(path, "rb") as f:
            return SigningKey.from_pem_content(f.read(), password)

    def save_pem(self, path: str, password: Optional[bytes] = None) -> "SigningKey":
        # Save this `PublicKey` instance to a PEM-encoded file.

        # Args:
        #     path: The path to save the key to.

        # Returns:
        #     This `PublicKey` instance.
        with open(path, "wb") as f:
            f.write(
                self._key.private_bytes(
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
        # Load a `SigningKey` instance from a PEM-encoded byte string.

        # Args:
        #     content: The PEM-encoded byte string to load the key from.
        #     password: The password to use to decrypt the key, if it is encrypted.

        # Returns:
        #     The `SigningKey` instance loaded from the given byte string.
        return SigningKey(serialization.load_pem_private_key(content, password))


class Identity:
    @staticmethod
    def create(
        name: Optional[str] = "bastionlab-identity", password: Optional[bytes] = None
    ) -> SigningKey:
        """
        Generate a new signing key with the given name and password.

        Args:
            name: The name to use for the signing key. If not provided, the default
                name "bastionlab-identity" will be used.
            password: The password to use to encrypt the signing key. If not provided,
                the key will not be encrypted.

        Returns:
            The generated signing key.
        """
        return SigningKey.keygen(name, password)

    @staticmethod
    def load(name: str) -> SigningKey:
        """
        Load a signing key with the given name.

        Args:
            name: The name of the signing key to load.

        Returns:
            The signing key with the given name.
        """
        return SigningKey.from_pem(name, None)
