use tch::TchError;

fn read_le_usize(input: &mut &[u8]) -> usize {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<usize>());
    *input = rest;
    usize::from_le_bytes(int_bytes.try_into().unwrap())
}

/// A buffer to serialize/deserialize byte data in the following format:
///
/// `obj = [length: 8 bytes, little-endian | data: length bytes]`
///
/// `stream = [obj, ...]`
#[derive(Debug)]
pub struct SizedObjectsBytes(Vec<u8>);

impl SizedObjectsBytes {
    /// returns a new empty `SizedObjectBytes` buffer.
    pub fn new() -> Self {
        SizedObjectsBytes(Vec::new())
    }

    /// Adds a new bytes object to the buffer, its size is automatically
    /// converted to little-endian bytes and prepended to the data.
    pub fn append_back(&mut self, mut bytes: Vec<u8>) {
        self.0.extend_from_slice(&bytes.len().to_le_bytes());
        self.0.append(&mut bytes);
    }

    /// Removes the eight first bytes of the buffer and interprets them
    /// as the little-endian bytes of the length. Then removes and returns
    /// the next length bytes from the buffer.
    pub fn remove_front(&mut self) -> Vec<u8> {
        let len = read_le_usize(&mut &self.0.drain(..8).collect::<Vec<u8>>()[..]);
        self.0.drain(..len).collect()
    }

    /// Get raw bytes.
    pub fn get(&self) -> &Vec<u8> {
        &self.0
    }
}

impl From<SizedObjectsBytes> for Vec<u8> {
    fn from(value: SizedObjectsBytes) -> Self {
        value.0
    }
}

impl From<Vec<u8>> for SizedObjectsBytes {
    fn from(value: Vec<u8>) -> Self {
        SizedObjectsBytes(value)
    }
}

impl Iterator for SizedObjectsBytes {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0.len() > 0 {
            Some(self.remove_front())
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct BinaryModule(pub(crate) Vec<u8>);

impl TryFrom<SizedObjectsBytes> for BinaryModule {
    type Error = TchError;

    fn try_from(mut value: SizedObjectsBytes) -> Result<Self, Self::Error> {
        let object = value.next().ok_or(TchError::FileFormat(String::from(
            "Invalid data, expected at least one object in stream.",
        )))?;
        Ok(BinaryModule(object))
    }
}
