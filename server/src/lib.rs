#![allow(dead_code)] // TODO: remove this when all the tests are implemented
mod authentication;

#[cfg(test)]
mod tests {
    #[test]
    fn test_keymanagement_loadfromdir_empty() {
        let res = crate::authentication::KeyManagement::load_from_dir("".to_string());
        assert!(res.is_err());
    }
    #[test]
    fn test_keymanagement_loadfromdir_wrongpath() {
        let res = crate::authentication::KeyManagement::load_from_dir(std::string::String::from(
            "wrongpath",
        ));
        assert!(res.is_err());
    }
}
