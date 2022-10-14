use bastionai_common::auth::AuthExtension;
use tch::TchError;
use tonic::Request;
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

pub fn fetch_username_and_userid<T>(
    request: &Request<T>,
    special_case: bool,
) -> Result<(Option<String>, Option<String>), Status> {
    let auth_ext = request.extensions().get::<AuthExtension>().cloned();

    if special_case && (auth_ext.is_none() || !auth_ext.as_ref().unwrap().is_logged()) {
        return Err(Status::permission_denied("You must be logged in"));
    }
    let userid = match auth_ext.as_ref() {
        Some(auth_ext) => match auth_ext.userid() {
            Some(id) => id.to_string().into(),
            None => None,
        },
        None => None,
    };

    let username = match auth_ext.as_ref() {
        Some(auth_ext) => match auth_ext.username() {
            Some(username) => username.into(),
            None => None,
        },
        None => None,
    };

    Ok((username, userid))
}
