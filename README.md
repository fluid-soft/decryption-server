# Decryption Server
Decryption server is used for encryption/decryption on dedicated inference servers in [Fluid Private API](https://getfluid.app/fluidpro). Decryption server generates private key and doesn't share it with anyone else. So if you're sending message signed by inference's server public key, only the server doing AI will be able to read and process it. 

# license
[GPL-3](LICENSE)