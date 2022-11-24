use tch::TchError;
use tonic::Status;

/// Converts a [`tch::TchError`]-based result into a [`tonic::Status`]-based one.
pub fn tcherror_to_status<T>(input: Result<T, TchError>) -> Result<T, Status> {
    input.map_err(|err| Status::internal(format!("Torch error: {}", err)))
}

/// Utility function for pretty printing
pub fn fill_blank_and_print(content: &str, size: usize) {
    let trail_char = "#";
    let trail: String = trail_char.repeat((size - 2 - content.len()) / 2);
    let trail2: String =
        trail_char.repeat(((size - 2 - content.len()) as f32 / 2.0).ceil() as usize);
    println!("{} {} {}", trail, content, trail2);
}
