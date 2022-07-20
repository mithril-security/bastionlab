# bastionai

# How we serialize/deserialize artifacts
- We either prepend or append the size of the artifact (either Tensors or Modules), in bytes to the artifact before sending it from the client.
- For deserialization, we extract first 4 bytes (size) and then deserialize the artifact.
- Likewise, for the serializing the artifact, we prepend the size of the artifact before sending back to the client.