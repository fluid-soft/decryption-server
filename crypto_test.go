package main

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"testing"
)

func TestGenerateNewKeyPair(t *testing.T) {
	keyPair, err := GenerateNewKeyPair()
	if err != nil {
		t.Fatalf("Failed to generate key pair: %v", err)
	}

	if keyPair.PrivateKey == ([56]byte{}) {
		t.Error("PrivateKey is empty")
	}
	if keyPair.PublicKey == ([56]byte{}) {
		t.Error("PublicKey is empty")
	}
	if keyPair.ExpiresAt.IsZero() {
		t.Error("ExpiresAt is zero")
	}
}

func TestGetEncodedPublicKey(t *testing.T) {
	keyPair, _ := GenerateNewKeyPair()
	encodedKey := GetEncodedPublicKey(keyPair.PublicKey)

	decodedKey, err := base64.StdEncoding.DecodeString(encodedKey)
	if err != nil {
		t.Fatalf("Failed to decode public key: %v", err)
	}

	if !bytes.Equal(keyPair.PublicKey[:], decodedKey) {
		t.Error("Decoded public key does not match original")
	}
}

func TestDeriveSharedSecret(t *testing.T) {
	aliceKeyPair, _ := GenerateNewKeyPair()
	bobKeyPair, _ := GenerateNewKeyPair()

	aliceSecret, err := DeriveSharedSecret(aliceKeyPair.PrivateKey, bobKeyPair.PublicKey[:])
	if err != nil {
		t.Fatalf("Alice failed to derive shared secret: %v", err)
	}

	bobSecret, err := DeriveSharedSecret(bobKeyPair.PrivateKey, aliceKeyPair.PublicKey[:])
	if err != nil {
		t.Fatalf("Bob failed to derive shared secret: %v", err)
	}

	if !bytes.Equal(aliceSecret, bobSecret) {
		t.Error("Derived shared secrets do not match")
	}
}

func TestAESEncryptionDecryption(t *testing.T) {
	key := make([]byte, 32) // 256-bit key
	_, err := rand.Read(key)
	if err != nil {
		t.Fatalf("Failed to generate random key: %v", err)
	}

	plaintext := []byte("Hello, World!")

	ciphertext, err := EncryptAES(plaintext, key)
	if err != nil {
		t.Fatalf("Encryption failed: %v", err)
	}

	decrypted, err := DecryptAES(ciphertext, key)
	if err != nil {
		t.Fatalf("Decryption failed: %v", err)
	}

	if !bytes.Equal(plaintext, decrypted) {
		t.Error("Decrypted text does not match original plaintext")
	}
}

func TestEndToEndEncryptionDecryption(t *testing.T) {
	// Generate key pairs for Alice and Bob
	aliceKeyPair, _ := GenerateNewKeyPair()
	bobKeyPair, _ := GenerateNewKeyPair()

	// Alice encrypts a message for Bob
	plaintext := []byte("Secret message from Alice to Bob")

	aliceSharedSecret, err := DeriveSharedSecret(aliceKeyPair.PrivateKey, bobKeyPair.PublicKey[:])
	if err != nil {
		t.Fatalf("Alice failed to derive shared secret: %v", err)
	}

	if len(aliceSharedSecret) == 0 {
		t.Fatal("Alice's derived shared secret is empty")
	}

	ciphertext, err := EncryptAES(plaintext, aliceSharedSecret)
	if err != nil {
		t.Fatalf("Alice failed to encrypt: %v", err)
	}

	// Bob decrypts the message from Alice
	bobSharedSecret, err := DeriveSharedSecret(bobKeyPair.PrivateKey, aliceKeyPair.PublicKey[:])
	if err != nil {
		t.Fatalf("Bob failed to derive shared secret: %v", err)
	}

	if len(bobSharedSecret) == 0 {
		t.Fatal("Bob's derived shared secret is empty")
	}

	decrypted, err := DecryptAES(ciphertext, bobSharedSecret)
	if err != nil {
		t.Fatalf("Bob failed to decrypt: %v", err)
	}

	if !bytes.Equal(plaintext, decrypted) {
		t.Error("Decrypted text does not match original plaintext")
	}
}

func TestCompressEncryptDecryptDecompress(t *testing.T) {
	key := make([]byte, 32) // 256-bit key
	_, err := rand.Read(key)
	if err != nil {
		t.Fatalf("Failed to generate random key: %v", err)
	}

	plaintext := []byte("Hello, World! This is a test of compression and encryption. " +
		"We're adding more text to ensure that the data is large enough to benefit from compression. " +
		"Repeated text like this is highly compressible. " +
		"The quick brown fox jumps over the lazy dog. " +
		"The quick brown fox jumps over the lazy dog. " +
		"The quick brown fox jumps over the lazy dog.")

	compressed, err := CompressAndEncryptAES(plaintext, key)
	if err != nil {
		t.Fatalf("Compression and encryption failed: %v", err)
	}

	decompressed, err := DecryptAndDecompressAES(compressed, key)
	if err != nil {
		t.Fatalf("Decryption and decompression failed: %v", err)
	}

	if !bytes.Equal(plaintext, decompressed) {
		t.Error("Decompressed and decrypted text does not match original plaintext")
	}

	// Check if compression actually reduced the size
	if len(compressed) >= len(plaintext) {
		t.Error("Compression did not reduce the size of the data")
	}
}

func TestEndToEndCompressionEncryptionDecryption(t *testing.T) {
	// Generate key pairs for Alice and Bob
	aliceKeyPair, _ := GenerateNewKeyPair()
	bobKeyPair, _ := GenerateNewKeyPair()

	// Alice compresses and encrypts a message for Bob
	plaintext := []byte("Secret message from Alice to Bob. This message is long enough to benefit from compression.")

	aliceSharedSecret, err := DeriveSharedSecret(aliceKeyPair.PrivateKey, bobKeyPair.PublicKey[:])
	if err != nil {
		t.Fatalf("Alice failed to derive shared secret: %v", err)
	}

	compressedAndEncrypted, err := CompressAndEncryptAES(plaintext, aliceSharedSecret)
	if err != nil {
		t.Fatalf("Alice failed to compress and encrypt: %v", err)
	}

	// Bob decrypts and decompresses the message from Alice
	bobSharedSecret, err := DeriveSharedSecret(bobKeyPair.PrivateKey, aliceKeyPair.PublicKey[:])
	if err != nil {
		t.Fatalf("Bob failed to derive shared secret: %v", err)
	}

	decryptedAndDecompressed, err := DecryptAndDecompressAES(compressedAndEncrypted, bobSharedSecret)
	if err != nil {
		t.Fatalf("Bob failed to decrypt and decompress: %v", err)
	}

	if !bytes.Equal(plaintext, decryptedAndDecompressed) {
		t.Error("Decrypted and decompressed text does not match original plaintext")
	}
}
