/**************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
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

#include <climits>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include <sys/random.h>
#include <iostream>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/ts.h>
#include <openssl/ecdh.h>
#include <openssl/bn.h>
#include <openssl/pem.h>
#include <openssl/sha.h>
#include <cstring>
#include "lib.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"

/**
 * Calculate the complete SHA256/SHA384 digest of the input message.
 * Use for RSA and ECDSA, not ECDH
 * Formerly called CalcHashDigest
 *
 * msg       : message buffer to hash.
 * msg_len   : length of the input message.
 *             - For SEV_CERTs, use PubKeyOffset (number of bytes to be hashed,
 *               from the top of the sev_cert until the first signature.
 *               Version through and including pub_key)
 * digest    : output buffer for the final digest.
 * digest_len: length of the output buffer.
 */
bool digest_sha(const void *msg, size_t msg_len, uint8_t *digest,
                size_t digest_len, SHA_TYPE sha_type)
{
    bool ret = false;

    do {    //TODO 384 vs 512 is all a mess
        if ((sha_type == SHA_TYPE_256 && digest_len != SHA256_DIGEST_LENGTH)/* ||
            (sha_type == SHA_TYPE_384 && digest_len != SHA384_DIGEST_LENGTH)*/)
                break;

        if (sha_type == SHA_TYPE_256) {
            SHA256_CTX context;

            if (SHA256_Init(&context) != 1)
                break;
            if (SHA256_Update(&context, (void *)msg, msg_len) != 1)
                break;
            if (SHA256_Final(digest, &context) != 1)
                break;
        }
        else if (sha_type == SHA_TYPE_384) {
            SHA512_CTX context;

            if (SHA384_Init(&context) != 1)
                break;
            if (SHA384_Update(&context, (void *)msg, msg_len) != 1)
                break;
            if (SHA384_Final(digest, &context) != 1)
                break;
        }

        ret = true;
    } while (0);

    return ret;
}


/**
 * It would be easier if we could just pass in the populated ECDSA_SIG from
 *  ecdsa_sign instead of using sev_sig to BigNums as the intermediary, but we
 *  do need to ecdsa_verify to verify something signed by firmware, so we
 *  wouldn't have the ECDSA_SIG
 */
bool ecdsa_verify(sev_sig *sig, EVP_PKEY **pub_evp_key, uint8_t *digest, size_t length)
{
    bool is_valid = false;
    EC_KEY *pub_ec_key = NULL;
    BIGNUM *r = NULL;
    BIGNUM *s = NULL;
    ECDSA_SIG *ecdsa_sig = NULL;

    do {
        pub_ec_key = EVP_PKEY_get1_EC_KEY(*pub_evp_key);
        if (!pub_ec_key)
            break;

        // Store the x and y components as separate BIGNUM objects. The values in the
        // SEV certificate are little-endian, must reverse bytes before storing in BIGNUM
        r = BN_lebin2bn(sig->ecdsa.r, sizeof(sig->ecdsa.r), NULL);  // New's up BigNum
        s = BN_lebin2bn(sig->ecdsa.s, sizeof(sig->ecdsa.s), NULL);
        
        // Create a ecdsa_sig from the bignums and store in sig
        ecdsa_sig = ECDSA_SIG_new();
        ECDSA_SIG_set0(ecdsa_sig, r, s);

        // Validation will also be done by the FW
        if (ECDSA_do_verify(digest, (uint32_t)length, ecdsa_sig, pub_ec_key) != 1) {
            ECDSA_SIG_free(ecdsa_sig);
            break;
        }
        ECDSA_SIG_free(ecdsa_sig);

        is_valid = true;
    } while (0);

    // Free memory
    EC_KEY_free(pub_ec_key);

    return is_valid;
}

/**
 * A generic sign function that takes a byte array (not specifically an sev_cert)
 *  and signs it using an sev_sig
 *
 * Note that verify always happens, even after a sign operation, just to make
 *  sure the sign worked correctly
 */
static bool sign_verify_message(sev_sig *sig, EVP_PKEY **evp_key_pair, const uint8_t *msg,
                                size_t length, const SEV_SIG_ALGO algo, bool sign)
{
    bool is_valid = false;
    hmac_sha_256 sha_digest_256;   // Hash on the cert from Version to PubKey
    hmac_sha_512 sha_digest_384;   // Hash on the cert from Version to PubKey
    SHA_TYPE sha_type;
    uint8_t *sha_digest = NULL;
    size_t sha_length;
    
    do {
        // Determine if SHA_TYPE is 256 bit or 384 bit
        if (algo == SEV_SIG_ALGO_RSA_SHA256 || algo == SEV_SIG_ALGO_ECDSA_SHA256 ||
            algo == SEV_SIG_ALGO_ECDH_SHA256)
        {
            sha_type = SHA_TYPE_256;
            sha_digest = sha_digest_256;
            sha_length = sizeof(hmac_sha_256);
        }
        else if (algo == SEV_SIG_ALGO_RSA_SHA384 || algo == SEV_SIG_ALGO_ECDSA_SHA384 ||
                 algo == SEV_SIG_ALGO_ECDH_SHA384)
        {
            sha_type = SHA_TYPE_384;
            sha_digest = sha_digest_384;
            sha_length = sizeof(hmac_sha_512);
        }
        else
        {
            break;
        }
        memset(sha_digest, 0, sha_length);

        // Calculate the hash digest
        if (!digest_sha(msg, length, sha_digest, sha_length, sha_type))
            break;

        if ((algo == SEV_SIG_ALGO_ECDSA_SHA256) || (algo == SEV_SIG_ALGO_ECDSA_SHA384)) {
            if (!ecdsa_verify(sig, evp_key_pair, sha_digest, sha_length))
                break;
        }
        else if ((algo == SEV_SIG_ALGO_ECDH_SHA256) || (algo == SEV_SIG_ALGO_ECDH_SHA384)) {
            printf("Error: ECDH signing unsupported");
            break;                       // Error unsupported
        }
        else {
            printf("Error: invalid signing algo. Can't sign");
            break;                          // Invalid params
        }

        is_valid = true;
    } while (0);

    return is_valid;
}


bool verify_message(sev_sig *sig, EVP_PKEY **evp_key_pair, const uint8_t *msg,
                    size_t length, const SEV_SIG_ALGO algo)
{
    return sign_verify_message(sig, evp_key_pair, msg, length, algo, false);
}


bool x509_validate_signature(X509 *child_cert, X509 *intermediate_cert, X509 *parent_cert)
{
    bool ret = false;
    X509_STORE *store = NULL;
    X509_STORE_CTX *store_ctx = NULL;

    do {
        // Create the store
        store = X509_STORE_new();
        if (!store)
            break;

        // Add the parent cert to the store
        if (X509_STORE_add_cert(store, parent_cert) != 1) {
            printf("Error adding parent_cert to x509_store\n");
            break;
        }

        // Add the intermediate cert to the store
        if (intermediate_cert) {
            if (X509_STORE_add_cert(store, intermediate_cert) != 1) {
                printf("Error adding intermediate_cert to x509_store\n");
                break;
            }
        }

        // Create the store context
        store_ctx = X509_STORE_CTX_new();
        if (!store_ctx) {
            printf("Error creating x509_store_context\n");
            break;
        }

        // Pass the store (parent and intermediate cert) and child cert (that we want to verify) into the store context
        if (X509_STORE_CTX_init(store_ctx, store, child_cert, NULL) != 1) {
            printf("Error initializing 509_store_context\n");
            break;
        }

        // Specify which cert to validate
        X509_STORE_CTX_set_cert(store_ctx, child_cert);

        // Verify the certificate
        ret = X509_verify_cert(store_ctx);

        // Print out error code
        if (ret == 0)
            printf("Error verifying cert: %s\n", X509_verify_cert_error_string(X509_STORE_CTX_get_error(store_ctx)));

        if (ret != 1)
            break;

        ret = true;
    } while (0);

    // Cleanup
    if (store_ctx)
        X509_STORE_CTX_free(store_ctx);
    if (store)
        X509_STORE_free(store);

    return ret;
}


int validate_cert_chain_vcek(void* ask_arg, void* ark_arg, X509* x509_vcek, int ask_len, int ark_len)
{
    int cmd_ret = 0;    //ERROR_UNSUPPORTED;
    
    X509 *x509_ask = NULL;
    X509 *x509_ark = NULL;

    BIO *bio_ark = NULL;
    BIO *bio_ask = NULL;

    do {
        
        // Load ARK certificate and pub_key
        bio_ark = BIO_new_mem_buf((void *)ark_arg,ark_len);
        PEM_read_bio_X509(bio_ark, &x509_ark, NULL, NULL);
        
        // Load ASK certificate and pub_key
        bio_ask = BIO_new_mem_buf((void *)ask_arg,ask_len);
        PEM_read_bio_X509(bio_ask, &x509_ask, NULL, NULL);
        

        // Verify the signatures of the certs
        if (!x509_validate_signature(x509_ark, NULL, x509_ark)) {   // Verify the ARK self-signed the ARK
            printf("Error validating signature of x509_ark certs\n");
            break;
        }

        if (!x509_validate_signature(x509_ask, NULL, x509_ark)) {   // Verify the ARK signed the ASK
            printf("Error validating signature of x509_ask certs\n");
            break;
        }

        if (!x509_validate_signature(x509_vcek, x509_ask, x509_ark)) {  // Verify the ASK signed the VCEK
            printf("Error validating signature of x509_vcek certs\n");
            break;
        }

        printf("VCEK cert chain validated successfully!\n");
        cmd_ret = 1;    //STATUS_SUCCESS;
    } while (0);

    X509_free(x509_vcek);
    X509_free(x509_ask);
    X509_free(x509_ark);

    return cmd_ret;
}


int attest(char report_arr[], char vcek_arr[], char ask_arr[], char ark_arr[], int vcek_len, int ask_len, int ark_len)
{
    void *report_arg = report_arr;
    void *vcek_arg = vcek_arr;
    void *ask_arg = ask_arr;   
    void *ark_arg = ark_arr;      
    
    int success = 1;
    EVP_PKEY *vcek_pub_key = NULL;
    X509 *x509_vcek = NULL;
    BIO *bio = NULL;

    do {
        
        snp_attestation_report_t *report = (snp_attestation_report_t *)report_arg;
        
        bio = BIO_new_mem_buf((void*)vcek_arg, vcek_len);
        PEM_read_bio_X509(bio,&x509_vcek, NULL, NULL);
        vcek_pub_key = X509_get_pubkey(x509_vcek);
        
        if (!vcek_pub_key)
            break;

        if(!validate_cert_chain_vcek(ask_arg, ark_arg, x509_vcek, ask_len, ark_len)) {
            break;
        }
        
        // Validate the report
        success = verify_message((sev_sig *)&report->signature,
                                  &vcek_pub_key, (uint8_t *)report_arg,
                                  offsetof(snp_attestation_report_t, signature),
                                  SEV_SIG_ALGO_ECDSA_SHA384);
        if (!success) {
            printf("Error: Guest report failed to validate\n");
            break;
        }

        printf("Guest report validated successfully!\n");
        success = 0;
        
    } while (0);

    // Free memory
    EVP_PKEY_free(vcek_pub_key);
    X509_free(x509_vcek);

    return success;
}

namespace py = pybind11;
PYBIND11_MODULE(_attestation_c, m) {
    m.doc() = "pybind11 attestation in C";

    m.def("attest", &attest, "Attestation function",
      py::arg("report"), py::arg("vcek"),py::arg("ask"), py::arg("ark"), py::arg("vcek_len"),py::arg("ask_len"), py::arg("ark_len"));
}
