import unittest

from bastionai import LicenseBuilder, Rule, SigningKey, PublicKey
import os

# print(SigningKey.generate().pubkey.pem)
pubkey_c = PublicKey.from_pem_content(
    b"""
-----BEGIN PUBLIC KEY-----
MFYwEAYHKoZIzj0CAQYFK4EEAAoDQgAEWrOcsEFyWycSZs4UV0yeqU5i5eqM3DDN
QHB+efLAsPSwVlECXop1cOAvEZ2rZ0aMBNH1430cOdA+5EEj+hjA3Q==
-----END PUBLIC KEY-----
"""
)
pubkey_c2 = PublicKey.from_pem_content(
    b"""
-----BEGIN PUBLIC KEY-----
MFYwEAYHKoZIzj0CAQYFK4EEAAoDQgAE7LSxyrZh+RCFk5XIzIjiNrMcZzk/7J7k
xVm+mIqYW+keO/GJqb1D/TaY8t/VIiG9uxXxcjLBx/uv4FMWaMGFBw==
-----END PUBLIC KEY-----
"""
)
pubkey_c3 = PublicKey.from_pem_content(
    b"""
-----BEGIN PUBLIC KEY-----
MFYwEAYHKoZIzj0CAQYFK4EEAAoDQgAEimbFNSfz0sGwY9s+1sO7nLRJL7Q4z4v9
g42G4S62ur1BbTtvCF9Y187XJT/miGrfARLse7IgAUZkwgRuPL2UnQ==
-----END PUBLIC KEY-----
"""
)


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
        with self.assertRaises(
            TypeError
        ):  # Password was not given but private key is encrypted
            SigningKey.from_pem("/tmp/hello.key.pem")
        with self.assertRaises(
            TypeError
        ):  # Password was not given but private key is encrypted
            SigningKey.from_pem_or_generate("/tmp/hello.key.pem")
        os.remove("/tmp/hello.key.pem")
        SigningKey.from_pem_or_generate("/tmp/hello.key.pem")
        with self.assertRaises(
            TypeError
        ):  # Password was given but private key is not encrypted.
            SigningKey.from_pem_or_generate("/tmp/hello.key.pem", b"123")
        SigningKey.from_pem("/tmp/hello.key.pem")
        os.remove("/tmp/hello.key.pem")
        with self.assertRaises(FileNotFoundError):  # No such file or directory
            SigningKey.from_pem("/tmp/hello.key.pem")
        SigningKey.generate().save_pem("/tmp/hello.key.pem")
        pk3 = SigningKey.from_pem("/tmp/hello.key.pem")
        os.remove("/tmp/hello.key.pem")

        # public key
        with self.assertRaises(FileNotFoundError):  # No such file or directory
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
        import cbor2
        print(cbor2.loads(LicenseBuilder.default_with_pubkey(pubkey_c).ser()))


        builder = (
            LicenseBuilder.default_with_pubkey(pubkey_c)
            .trainable(with_dataset=b"hash1")
            .deletable(signed_with=pubkey_c2)
        )

        self.assertEqual(
            builder.__str__(),
            """\
License {
  train=AtLeastNOf(1, [SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd), WithDataset(6861736831)]),
  train_metric=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  test=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  test_metric=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  list=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  fetch=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  delete=AtLeastNOf(1, [SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd), SignedWith(3056301006072a8648ce3d020106052b8104000a03420004ecb4b1cab661f910859395c8cc88e236b31c67393fec9ee4c559be988a985be91e3bf189a9bd43fd3698f2dfd52221bdbb15f17232c1c7fbafe0531668c18507)]),
  result_strategy=And,
}""",
        )

        builder = (
            LicenseBuilder.default_with_pubkey(pubkey_c)
            .trainable(either=[{"signed_with": pubkey_c2}, {"signed_with": pubkey_c3}])
            .trainable(Rule(signed_with=pubkey_c2))
            .trainable(either=[Rule(with_checkpoint="bcbdbd")])
            .created_checkpoints_license(get_from_checkpoint=True)
        )

        self.assertEqual(
            builder.__str__(),
            """\
License {
  train=AtLeastNOf(1, [SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd), SignedWith(3056301006072a8648ce3d020106052b8104000a03420004ecb4b1cab661f910859395c8cc88e236b31c67393fec9ee4c559be988a985be91e3bf189a9bd43fd3698f2dfd52221bdbb15f17232c1c7fbafe0531668c18507), SignedWith(3056301006072a8648ce3d020106052b8104000a034200048a66c53527f3d2c1b063db3ed6c3bb9cb4492fb438cf8bfd838d86e12eb6babd416d3b6f085f58d7ced7253fe6886adf0112ec7bb220014664c2046e3cbd949d), SignedWith(3056301006072a8648ce3d020106052b8104000a03420004ecb4b1cab661f910859395c8cc88e236b31c67393fec9ee4c559be988a985be91e3bf189a9bd43fd3698f2dfd52221bdbb15f17232c1c7fbafe0531668c18507), WithCheckpoint(bcbdbd)]),
  train_metric=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  test=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  test_metric=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  list=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  fetch=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  delete=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  result_strategy=Checkpoint,
}""",
        )

        builder = LicenseBuilder.default_with_pubkey(
            pubkey_c
        ).created_checkpoints_license(
            use_license=LicenseBuilder.default_with_pubkey(pubkey_c2)
        )

        self.assertEqual(
            builder.__str__(),
            """\
License {
  train=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  train_metric=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  test=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  test_metric=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  list=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  fetch=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  delete=SignedWith(3056301006072a8648ce3d020106052b8104000a034200045ab39cb041725b271266ce14574c9ea94e62e5ea8cdc30cd40707e79f2c0b0f4b05651025e8a7570e02f119dab67468c04d1f5e37d1c39d03ee44123fa18c0dd),
  result_strategy=Custom(License {
    train=SignedWith(3056301006072a8648ce3d020106052b8104000a03420004ecb4b1cab661f910859395c8cc88e236b31c67393fec9ee4c559be988a985be91e3bf189a9bd43fd3698f2dfd52221bdbb15f17232c1c7fbafe0531668c18507),
    train_metric=SignedWith(3056301006072a8648ce3d020106052b8104000a03420004ecb4b1cab661f910859395c8cc88e236b31c67393fec9ee4c559be988a985be91e3bf189a9bd43fd3698f2dfd52221bdbb15f17232c1c7fbafe0531668c18507),
    test=SignedWith(3056301006072a8648ce3d020106052b8104000a03420004ecb4b1cab661f910859395c8cc88e236b31c67393fec9ee4c559be988a985be91e3bf189a9bd43fd3698f2dfd52221bdbb15f17232c1c7fbafe0531668c18507),
    test_metric=SignedWith(3056301006072a8648ce3d020106052b8104000a03420004ecb4b1cab661f910859395c8cc88e236b31c67393fec9ee4c559be988a985be91e3bf189a9bd43fd3698f2dfd52221bdbb15f17232c1c7fbafe0531668c18507),
    list=SignedWith(3056301006072a8648ce3d020106052b8104000a03420004ecb4b1cab661f910859395c8cc88e236b31c67393fec9ee4c559be988a985be91e3bf189a9bd43fd3698f2dfd52221bdbb15f17232c1c7fbafe0531668c18507),
    fetch=SignedWith(3056301006072a8648ce3d020106052b8104000a03420004ecb4b1cab661f910859395c8cc88e236b31c67393fec9ee4c559be988a985be91e3bf189a9bd43fd3698f2dfd52221bdbb15f17232c1c7fbafe0531668c18507),
    delete=SignedWith(3056301006072a8648ce3d020106052b8104000a03420004ecb4b1cab661f910859395c8cc88e236b31c67393fec9ee4c559be988a985be91e3bf189a9bd43fd3698f2dfd52221bdbb15f17232c1c7fbafe0531668c18507),
    result_strategy=And,
  }),
}""",
        )

        serialized = builder.ser()