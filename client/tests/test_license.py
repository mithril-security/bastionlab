import unittest

from bastionai import LicenseBuilder, Rule, SigningKey, PublicKey
import os

pubkey = "pubkeywow"


class License(unittest.TestCase):
    def test_privkey(self):
        try:
            os.remove("/tmp/hello.key.pem")
        except FileNotFoundError:
            pass
        try:
            os.remove("/tmp/hello.pem")
        except FileNotFoundError:
            pass

        # private key
        pk1 = SigningKey.from_pem_or_generate("/tmp/hello.key.pem", b"123")
        pk2 = SigningKey.from_pem("/tmp/hello.key.pem", b"123")
        with self.assertRaises(TypeError): # Password was not given but private key is encrypted
            SigningKey.from_pem("/tmp/hello.key.pem")
        with self.assertRaises(TypeError): # Password was not given but private key is encrypted
            SigningKey.from_pem_or_generate("/tmp/hello.key.pem")
        os.remove("/tmp/hello.key.pem")
        SigningKey.from_pem_or_generate("/tmp/hello.key.pem")
        with self.assertRaises(TypeError): # Password was given but private key is not encrypted.
            SigningKey.from_pem_or_generate("/tmp/hello.key.pem", b"123")
        SigningKey.from_pem("/tmp/hello.key.pem")
        os.remove("/tmp/hello.key.pem")
        with self.assertRaises(FileNotFoundError): # No such file or directory
            SigningKey.from_pem("/tmp/hello.key.pem")
        SigningKey.generate().save_pem("/tmp/hello.key.pem")
        pk3 = SigningKey.from_pem("/tmp/hello.key.pem")
        os.remove("/tmp/hello.key.pem")

        # public key
        with self.assertRaises(FileNotFoundError): # No such file or directory
            PublicKey.from_pem("/tmp/hello.pem")
        pk1.pubkey.save_pem("/tmp/hello.pem")
        pub1 = PublicKey.from_pem("/tmp/hello.pem")
        pub2 = pk2.pubkey
        pub3 = pk3.pubkey
        pub4 = pk1.pubkey
        self.assertEqual(pub1.bytes, pub2.bytes)
        self.assertEqual(pub1.hash, pub2.hash)
        self.assertEqual(pub1.bytes, pub4.bytes)
        self.assertEqual(pub1.hash, pub4.hash)
        self.assertNotEqual(pub3.bytes, pub2.bytes)
        self.assertNotEqual(pub3.hash, pub2.hash)
        self.assertNotEqual(pub3.bytes, pub4.bytes)
        self.assertNotEqual(pub3.hash, pub4.hash)
        os.remove("/tmp/hello.pem")

        pass

    def test_create(self):
        builder = (
            LicenseBuilder.default_with_pubkey(pubkey)
            .trainable(with_dataset="hash1")
            .deletable(signed_with="pubkey2")
        )

        assert (
            builder.__str__()
            == """\
License {
  train=AtLeastNOf(1, [SignedWith(pubkeywow), WithDataset(hash1)]),
  train_metric=SignedWith(pubkeywow),
  test=SignedWith(pubkeywow),
  test_metric=SignedWith(pubkeywow),
  list=SignedWith(pubkeywow),
  fetch=SignedWith(pubkeywow),
  delete=AtLeastNOf(1, [SignedWith(pubkeywow), SignedWith(pubkey2)]),
  result_strategy=And,
}"""
        )

        builder = (
            LicenseBuilder.default_with_pubkey(pubkey)
            .trainable(
                either=[{"signed_with": "pubkey2"}, {"signed_with": "pubkey3"}]
            )
            .trainable(Rule(signed_with="pubkey4"))
            .trainable(either=[Rule(with_checkpoint="hashhh")])
            .created_checkpoints_license(get_from_checkpoint=True)
        )

        self.assertEqual(
            builder.__str__(),
            """\
License {
  train=AtLeastNOf(1, [SignedWith(pubkeywow), SignedWith(pubkey2), SignedWith(pubkey3), SignedWith(pubkey4), WithCheckpoint(hashhh)]),
  train_metric=SignedWith(pubkeywow),
  test=SignedWith(pubkeywow),
  test_metric=SignedWith(pubkeywow),
  list=SignedWith(pubkeywow),
  fetch=SignedWith(pubkeywow),
  delete=SignedWith(pubkeywow),
  result_strategy=Checkpoint,
}""",
        )

        builder = LicenseBuilder.default_with_pubkey(
            pubkey
        ).created_checkpoints_license(
            use_license=LicenseBuilder.default_with_pubkey("pubkey2")
        )

        self.assertEqual(
            builder.__str__(),
            """\
License {
  train=SignedWith(pubkeywow),
  train_metric=SignedWith(pubkeywow),
  test=SignedWith(pubkeywow),
  test_metric=SignedWith(pubkeywow),
  list=SignedWith(pubkeywow),
  fetch=SignedWith(pubkeywow),
  delete=SignedWith(pubkeywow),
  result_strategy=Custom(License {
    train=SignedWith(pubkey2),
    train_metric=SignedWith(pubkey2),
    test=SignedWith(pubkey2),
    test_metric=SignedWith(pubkey2),
    list=SignedWith(pubkey2),
    fetch=SignedWith(pubkey2),
    delete=SignedWith(pubkey2),
    result_strategy=And,
  }),
}""",
        )

        serialized = builder.ser()
