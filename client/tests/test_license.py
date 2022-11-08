import unittest

from bastionai import LicenseBuilder, RuleBuilder, SigningKey, PublicKey, License
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


class LicenseTest(unittest.TestCase):
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
        builder = (
            LicenseBuilder.default_with_pubkey(pubkey_c)
            .trainable(with_dataset=b"hash1")
            .deletable(signed_with=pubkey_c2)
        )

        self.assertEqual(
            builder.__str__(),
            "License { "
            + "train=Rule(at_least_n_of(1, [Rule(signed_with(hash=bd9429166ae1b2cf019fb0f51ddbd69dc7ed3ccad5345696fa584d480f75bfc7)), Rule(with_dataset(6861736831))])), "
            + "test=Rule(signed_with(hash=bd9429166ae1b2cf019fb0f51ddbd69dc7ed3ccad5345696fa584d480f75bfc7)), "
            + "list=Rule(signed_with(hash=bd9429166ae1b2cf019fb0f51ddbd69dc7ed3ccad5345696fa584d480f75bfc7)), "
            + "fetch=Rule(signed_with(hash=bd9429166ae1b2cf019fb0f51ddbd69dc7ed3ccad5345696fa584d480f75bfc7)), "
            + "delete=Rule(at_least_n_of(1, [Rule(signed_with(hash=bd9429166ae1b2cf019fb0f51ddbd69dc7ed3ccad5345696fa584d480f75bfc7)), "
            + "Rule(signed_with(hash=528b9a636d6145c9ddcd0d3c70a11a30bf4fb535a8834a42c452630dde0b3e0a))])), "
            + "result_strategy=ResultStrategy(strategy=ResultStrategyKind.And) "
            + "}",
        )
        self.assertEqual(builder.build().ser(), License.deser(builder.build().ser()).ser())

        builder = (
            LicenseBuilder.default_with_pubkey(pubkey_c)
            .trainable(either=[{"signed_with": pubkey_c2}, {"signed_with": pubkey_c3}])
            .trainable(RuleBuilder(signed_with=pubkey_c2))
            .trainable(either=[RuleBuilder(with_checkpoint="bcbdbd")])
            .created_checkpoints_license(get_from_checkpoint=True)
        )

        self.assertEqual(
            builder.__str__(),
            "License { "
            + "train=Rule(at_least_n_of(1, [Rule(signed_with(hash=bd9429166ae1b2cf019fb0f51ddbd69dc7ed3ccad5345696fa584d480f75bfc7)), Rule(signed_with(hash=528b9a636d6145c9ddcd0d3c70a11a30bf4fb535a8834a42c452630dde0b3e0a)), Rule(signed_with(hash=202420a4d277b17d82ad4a94425b5625b77c26fcb173da916609b19b30b8a4c6)), Rule(signed_with(hash=528b9a636d6145c9ddcd0d3c70a11a30bf4fb535a8834a42c452630dde0b3e0a)), Rule(with_checkpoint(bcbdbd))])), "
            + "test=Rule(signed_with(hash=bd9429166ae1b2cf019fb0f51ddbd69dc7ed3ccad5345696fa584d480f75bfc7)), "
            + "list=Rule(signed_with(hash=bd9429166ae1b2cf019fb0f51ddbd69dc7ed3ccad5345696fa584d480f75bfc7)), "
            + "fetch=Rule(signed_with(hash=bd9429166ae1b2cf019fb0f51ddbd69dc7ed3ccad5345696fa584d480f75bfc7)), "
            + "delete=Rule(signed_with(hash=bd9429166ae1b2cf019fb0f51ddbd69dc7ed3ccad5345696fa584d480f75bfc7)), "
            + "result_strategy=ResultStrategy(strategy=ResultStrategyKind.Checkpoint) "
            + "}",
        )
        self.assertEqual(builder.build().ser(), License.deser(builder.build().ser()).ser())

        builder = LicenseBuilder.default_with_pubkey(
            pubkey_c
        ).created_checkpoints_license(
            use_license=LicenseBuilder.default_with_pubkey(pubkey_c2)
        )

        self.assertEqual(
            builder.__str__(),
            "License { "
            + "train=Rule(signed_with(hash=bd9429166ae1b2cf019fb0f51ddbd69dc7ed3ccad5345696fa584d480f75bfc7)), "
            + "test=Rule(signed_with(hash=bd9429166ae1b2cf019fb0f51ddbd69dc7ed3ccad5345696fa584d480f75bfc7)), "
            + "list=Rule(signed_with(hash=bd9429166ae1b2cf019fb0f51ddbd69dc7ed3ccad5345696fa584d480f75bfc7)), "
            + "fetch=Rule(signed_with(hash=bd9429166ae1b2cf019fb0f51ddbd69dc7ed3ccad5345696fa584d480f75bfc7)), "
            + "delete=Rule(signed_with(hash=bd9429166ae1b2cf019fb0f51ddbd69dc7ed3ccad5345696fa584d480f75bfc7)), "
            + "result_strategy=ResultStrategy(strategy=ResultStrategyKind.Custom, "
            + "custom_license=License { "
            + "train=Rule(signed_with(hash=528b9a636d6145c9ddcd0d3c70a11a30bf4fb535a8834a42c452630dde0b3e0a)), "
            + "test=Rule(signed_with(hash=528b9a636d6145c9ddcd0d3c70a11a30bf4fb535a8834a42c452630dde0b3e0a)), "
            + "list=Rule(signed_with(hash=528b9a636d6145c9ddcd0d3c70a11a30bf4fb535a8834a42c452630dde0b3e0a)), "
            + "fetch=Rule(signed_with(hash=528b9a636d6145c9ddcd0d3c70a11a30bf4fb535a8834a42c452630dde0b3e0a)), "
            + "delete=Rule(signed_with(hash=528b9a636d6145c9ddcd0d3c70a11a30bf4fb535a8834a42c452630dde0b3e0a)), "
            + "result_strategy=ResultStrategy(strategy=ResultStrategyKind.And) "
            + "}) "
            + "}",
        )
        self.assertEqual(builder.build().ser(), License.deser(builder.build().ser()).ser())
