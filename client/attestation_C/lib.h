

/**************************************************************************
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **************************************************************************/


#include <string.h>
#include <stdint.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <cstring>                  // memset
#include <stdio.h>
#include <stdexcept>
#include <stdio.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/rsa.h>
#include <openssl/sha.h>

typedef union snp_tcb_version    // TCB
{
    struct
    {
        uint8_t boot_loader;    // SVN of PSP bootloader
        uint8_t tee;            // SVN of PSP operating system
        uint8_t reserved[4];
        uint8_t snp;            // SVN of SNP firmware
        uint8_t microcode;      // Lowest current patch level of all the cores
    } __attribute__((packed)) f;
    uint64_t val;
} __attribute__((packed)) snp_tcb_version_t;



typedef struct snp_attestation_report_platform_info
{
    uint32_t smt_en   : 1;
    uint64_t reserved : 63;
} __attribute__((packed)) snp_platform_info_t;

#define SNP_GMSG_MAX_REPORT_VERSION 1
typedef struct snp_attestation_report
{
    uint32_t version;               /* 0h */
    uint32_t guest_svn;             /* 4h */
    uint64_t policy;                /* 8h */
    uint8_t family_id[16];          /* 10h */
    uint8_t image_id[16];           /* 20h */
    uint32_t vmpl;                  /* 30h */
    uint32_t signature_algo;        /* 34h */
    snp_tcb_version_t tcb_version;  /* 38h */
    snp_platform_info_t platform_info; /* 40h */
    uint32_t author_key_en : 1;     /* 48h */
    uint32_t reserved      : 31;
    uint32_t reserved2;             /* 4C */
    uint8_t report_data[64];        /* 50h */
    uint8_t measurement[48];        /* 90h */
    uint8_t host_data[32];          /* C0h */
    uint8_t id_key_digest[48];      /* E0h */
    uint8_t author_key_digest[48];  /* 110h */
    uint8_t report_id[32];          /* 140h */
    uint8_t report_id_ma[32];       /* 160h */
    snp_tcb_version_t reported_tcb; /* 180h */
    uint8_t reserved3[0x1A0-0x188]; /* 188h-19Fh */
    uint8_t chip_id[64];            /* 1A0h */
    uint64_t committed_tcb;         /* 1E0h */
    uint8_t current_build;          /* 1E8h */
    uint8_t current_minor;          /* 1E9h */
    uint8_t current_major;          /* 1EAh */
    uint8_t reserved4;              /* 1EBh */
    uint8_t committed_build;         /* 1ECh */
    uint8_t committed_minor;         /* 1EDh */
    uint8_t committed_major;         /* 1EEh */
    uint8_t reserved5;              /* 1EFh */
    uint64_t launch_tcb;            /* 1F0h */
    uint8_t reserved6[0x2A0-0x1F8];  /* 1F8h-29Fh */
    uint8_t signature[0x4A0-0x2A0]; /* 2A0h-49Fh */
} __attribute__((packed)) snp_attestation_report_t;


#define PADDR_INVALID  ~(0x0ull)            /* -1 */


#if __cplusplus
typedef bool _Bool;
#endif

 
// ------------------------------------------------------------ //
// --- Definition of API-defined Encryption and HMAC values --- //
// ------------------------------------------------------------ //


// Chapter 2 - Summary of Keys
typedef uint8_t aes_128_key[128/8];
typedef uint8_t hmac_key_128[128/8];
typedef uint8_t hmac_sha_256[256/8];  // 256
typedef uint8_t hmac_sha_512[512/8];  // 384, 512
typedef uint8_t nonce_128[128/8];
typedef uint8_t iv_128[128/8];

// -------------------------------------------------------------------------- //
// -- Definition of API-defined Public Key Infrastructure (PKI) structures -- //
// -------------------------------------------------------------------------- //

// Appendix C.3: SEV Certificates
#define SEV_RSA_PUB_KEY_MAX_BITS    4096
#define SEV_ECDSA_PUB_KEY_MAX_BITS  576
#define SEV_ECDH_PUB_KEY_MAX_BITS   576
#define SEV_PUB_KEY_SIZE            (SEV_RSA_PUB_KEY_MAX_BITS/8)

// Appendix C.3.1 Public Key Formats - RSA Public Key
/**
 * SEV RSA Public key information.
 *
 * @modulus_size - Size of modulus in bits.
 * @pub_exp      - The public exponent of the public key.
 * @modulus      - The modulus of the public key.
 */
typedef struct __attribute__ ((__packed__)) sev_rsa_pub_key_t
{
    uint32_t    modulus_size;
    uint8_t     pub_exp[SEV_RSA_PUB_KEY_MAX_BITS/8];
    uint8_t     modulus[SEV_RSA_PUB_KEY_MAX_BITS/8];
} sev_rsa_pub_key;

/**
 * SEV Elliptical Curve algorithm details.
 *
 * @SEV_EC_INVALID - Invalid cipher size selected.
 * @SEV_EC_P256    - 256 bit elliptical curve cipher.
 * @SEV_EC_P384    - 384 bit elliptical curve cipher.
 */
typedef enum __attribute__((mode(QI))) SEV_EC
{
    SEV_EC_INVALID = 0,
    SEV_EC_P256    = 1,
    SEV_EC_P384    = 2,
} SEV_EC;

// Appendix C.3.2: Public Key Formats - ECDSA Public Key
/**
 * SEV Elliptical Curve DSA algorithm details.
 *
 * @curve - The SEV Elliptical curve ID.
 * @qx    - x component of the public point Q.
 * @qy    - y component of the public point Q.
 * @rmbz  - RESERVED. Must be zero!
 */
typedef struct __attribute__ ((__packed__)) sev_ecdsa_pub_key_t
{
    uint32_t    curve;      // SEV_EC as a uint32_t
    uint8_t     qx[SEV_ECDSA_PUB_KEY_MAX_BITS/8];
    uint8_t     qy[SEV_ECDSA_PUB_KEY_MAX_BITS/8];
    uint8_t     rmbz[SEV_PUB_KEY_SIZE-2*SEV_ECDSA_PUB_KEY_MAX_BITS/8-sizeof(uint32_t)];
} sev_ecdsa_pub_key;

// Appendix C.4: Signature Formats
/**
 * SEV Signature may be RSA or ECDSA.
 */
#define SEV_RSA_SIG_MAX_BITS        4096
#define SEV_ECDSA_SIG_COMP_MAX_BITS 576
#define SEV_SIG_SIZE                (SEV_RSA_SIG_MAX_BITS/8)

// Appendix C.4.1: RSA Signature
/**
 * SEV RSA Signature data.
 *
 * @S - Signature bits.
 */
typedef struct __attribute__ ((__packed__)) sev_rsa_sig_t
{
    uint8_t     s[SEV_RSA_SIG_MAX_BITS/8];
} sev_rsa_sig;

// Appendix C.4.2: ECDSA Signature
/**
 * SEV Elliptical Curve Signature data.
 *
 * @r    - R component of the signature.
 * @s    - S component of the signature.
 * @rmbz - RESERVED. Must be zero!
 */
typedef struct __attribute__ ((__packed__)) sev_ecdsa_sig_t
{
    uint8_t     r[SEV_ECDSA_SIG_COMP_MAX_BITS/8];
    uint8_t     s[SEV_ECDSA_SIG_COMP_MAX_BITS/8];
    uint8_t     rmbz[SEV_SIG_SIZE-2*SEV_ECDSA_SIG_COMP_MAX_BITS/8];
} sev_ecdsa_sig;

/**
 * SEV Signature may be RSA or ECDSA.
 */
typedef union
{
    sev_rsa_sig     rsa;
    sev_ecdsa_sig   ecdsa;
} sev_sig;

// Appendix C.1: ALGO Enumeration
/**
 * SEV Algorithm cipher codes.
 */
typedef enum __attribute__((mode(HI))) SEV_SIG_ALGO
{
    SEV_SIG_ALGO_INVALID      = 0x0,
    SEV_SIG_ALGO_RSA_SHA256   = 0x1,
    SEV_SIG_ALGO_ECDSA_SHA256 = 0x2,
    SEV_SIG_ALGO_ECDH_SHA256  = 0x3,
    SEV_SIG_ALGO_RSA_SHA384   = 0x101,
    SEV_SIG_ALGO_ECDSA_SHA384 = 0x102,
    SEV_SIG_ALGO_ECDH_SHA384  = 0x103,
} SEV_SIG_ALGO;

#define SEV_CERT_MAX_VERSION    1       // Max supported version
#define SEV_CERT_MAX_SIGNATURES 2       // Max number of sig's



/**
 * DIGEST
 */
#define DIGEST_SHA256_SIZE_BYTES    (256/8) // 32
#define DIGEST_SHA384_SIZE_BYTES    (384/8) // 48
#define DIGEST_SHA512_SIZE_BYTES    (512/8) // 64
typedef uint8_t DIGESTSHA256[DIGEST_SHA256_SIZE_BYTES];
typedef uint8_t DIGESTSHA384[DIGEST_SHA384_SIZE_BYTES];
typedef uint8_t DIGESTSHA512[DIGEST_SHA512_SIZE_BYTES];


typedef enum __attribute__((mode(QI))) SHA_TYPE
{
    SHA_TYPE_256 = 0,
    SHA_TYPE_384 = 1,
} SHA_TYPE;



bool digest_sha(const void *msg, size_t msg_len, uint8_t *digest,
                size_t digest_len, SHA_TYPE sha_type);

bool ecdsa_verify(sev_sig *sig, EVP_PKEY **pub_evp_key, uint8_t *digest, size_t length);

bool sign_message(sev_sig *sig, EVP_PKEY **evp_key_pair, const uint8_t *msg,
                  size_t length, const SEV_SIG_ALGO algo);

bool verify_message(sev_sig *sig, EVP_PKEY **evp_key_pair, const uint8_t *msg,
                    size_t length, const SEV_SIG_ALGO algo);

int attest(char report_arr[], char vcek_arr[], char ask_arr[], char ark_arr[],  int vcek_len, int ask_len, int ark_len);