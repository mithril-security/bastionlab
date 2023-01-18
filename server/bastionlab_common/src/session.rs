use std::collections::HashSet;
use std::net::SocketAddr;
use std::time::{Duration, SystemTime};

use bytes::Bytes;
use prost::Message;
use tonic::metadata::KeyRef;
use tonic::{Request, Response, Status};

use crate::auth::KeyManagement;
use crate::session_proto::{ClientInfo, SessionInfo};
use crate::{prelude::*, session_proto};

fn get_message<T: Message>(
    method: &[u8],
    req: &Request<T>,
    challenge: Bytes,
) -> Result<Vec<u8>, Status> {
    let mut res =
        Vec::with_capacity(method.len() + challenge.as_ref().len() + req.get_ref().encoded_len());
    res.extend_from_slice(method);
    res.extend_from_slice(challenge.as_ref());
    req.get_ref()
        .encode(&mut res)
        .map_err(|e| Status::internal(format!("error while encoding the request: {:?}", e)))?;
    Ok(res)
}

#[derive(Debug)]
pub struct Session {
    pub pubkey: String,
    pub user_ip: SocketAddr,
    pub expiry: SystemTime,
    pub client_info: ClientInfo,
}

#[derive(Debug)]
pub struct SessionManager {
    keys: Option<Mutex<KeyManagement>>,
    sessions: Arc<RwLock<HashMap<[u8; 32], Session>>>,
    session_expiry: u64,
    challenges: Mutex<HashSet<[u8; 32]>>,
}

impl SessionManager {
    pub fn new(keys: Option<KeyManagement>, session_expiry: u64) -> Self {
        Self {
            keys: keys.map(Mutex::new),
            sessions: Default::default(),
            session_expiry,
            challenges: Default::default(),
        }
    }

    pub fn auth_enabled(&self) -> bool {
        self.keys.is_some()
    }

    /// Get the user pubkey hash from a token
    pub fn get_user_id(&self, token: Option<Bytes>) -> Result<String, Status> {
        let token_bytes = match &token {
            Some(v) => &v[..],
            None => &[0u8; 32],
        };
        let sessions = self.sessions.read().expect("Poisoned lock");
        let session = sessions
            .get(token_bytes)
            .ok_or_else(|| Status::aborted("Session not found!"))?;

        let user_id = session.pubkey.clone();
        Ok(user_id)
    }

    /// Verify a request, return the user id token.
    pub fn verify_request<T>(&self, req: &Request<T>) -> Result<Option<Bytes>, Status> {
        if !self.auth_enabled() {
            return Ok(None);
        }
        let meta = req
            .metadata()
            .get_bin("accesstoken-bin")
            .ok_or_else(|| Status::invalid_argument("No access token in request metadata"))?;

        let access_token = meta
            .to_bytes()
            .map_err(|_| Status::invalid_argument("Could not decode accesstoken"))?;

        let mut tokens = self.sessions.write().expect("Poisoned lock");

        let session = tokens
            .get(access_token.as_ref())
            .ok_or_else(|| Status::aborted("Session not found!"))?;

        let recv_ip = &req
            .remote_addr()
            .ok_or_else(|| Status::aborted("User IP unavailable"))?;

        // ip verification
        if session.user_ip.ip() != recv_ip.ip() {
            return Err(Status::aborted("Unknown IP Address!"));
        }

        // expiry verification
        let curr_time = SystemTime::now();
        if curr_time > session.expiry {
            tokens.remove(access_token.as_ref());
            return Err(Status::aborted("Session Expired"));
        }

        Ok(Some(access_token))
    }

    /// When no token is given and auth is disabled, this will give the ClientInfo of the last
    /// session created
    pub fn get_client_info(&self, token: Option<Bytes>) -> Result<ClientInfo, Status> {
        let sessions = self.sessions.write().expect("Poisoned lock");
        let token = match &token {
            Some(v) => &v[..],
            None => &[0u8; 32],
        };
        let session = sessions
            .get(token)
            .ok_or_else(|| Status::aborted("Session not found!"))?;
        Ok(session.client_info.clone())
    }

    fn new_challenge(&self) -> [u8; 32] {
        let rng = ring::rand::SystemRandom::new();
        loop {
            if let Ok(challenge) = ring::rand::generate(&rng) {
                let challenge: [u8; 32] = challenge.expose();
                let mut lock = self.challenges.lock().expect("Poisoned lock");
                lock.insert(challenge);
                return challenge;
            }
        }
    }

    fn check_challenge<T: Message>(&self, request: &Request<T>) -> Result<Bytes, Status> {
        let challenge = request.metadata()
            .get_bin("challenge-bin")
            .ok_or_else(|| Status::unauthenticated("You must be authenticated to perform this action. Please reconnect with an identity."))?;
        let challenge_bytes = challenge.to_bytes().map_err(|_| {
            Status::invalid_argument(format!("Could not decode challenge {:?}", challenge))
        })?;
        let challenge = challenge_bytes.as_ref();
        {
            let mut lock = self.challenges.lock().expect("Poisoned lock");
            if !lock.remove(challenge) {
                return Err(Status::permission_denied("Challenge not found!"));
            }
        }

        Ok(challenge_bytes)
    }

    pub fn verify_if_owner(&self, public_hash: &str) -> Result<bool, Status> {
        if self.auth_enabled() == false {
            return Err(Status::permission_denied(
                "This operation requires authentication to be enabled.",
            ));
        }
        let keys_lock = self.keys.as_ref().map(|l| l.lock().expect("Poisoned lock"));
        if let Some(ref keys) = keys_lock {
            return Ok(keys.verify_owner(public_hash));
        }
        return Ok(false);
    }

    // TODO: move grpc specific things to the grpc service and not the session manager
    fn create_session(&self, request: Request<ClientInfo>) -> Result<SessionInfo, Status> {
        let user_ip = request
            .remote_addr()
            .ok_or_else(|| Status::aborted("Could not fetch IP Address from request"))?;
        let mut sessions = self.sessions.write().unwrap();

        if !self.auth_enabled() {
            // auth disabled
            let (token, expiry) = ([0u8; 32], SystemTime::now());
            sessions.insert(
                token.clone(),
                Session {
                    pubkey: "UNAUTHENTICATED".to_string(),
                    user_ip,
                    expiry,
                    client_info: request.into_inner(),
                },
            );
            return Ok(SessionInfo {
                token: token.to_vec(),
                expiry_time: self.session_expiry * 1000,
            });
        }
        // auth enabled

        // unwrap: self.keys is not None since auth is enabled
        let keys_lock = self.keys.as_ref().unwrap().lock().expect("Poisoned lock");

        let challenge = self.check_challenge(&request)?;

        // stripped key hash from the request metadata
        let pubkey_hash = request
            .metadata()
            .keys()
            .filter_map(|k| match k {
                KeyRef::Binary(key) => Some(key),
                _ => None,
            })
            .filter_map(|k| {
                let s = k.as_str().strip_suffix("-bin")?;
                s.strip_prefix("signature-")
            })
            // take only the first one
            .next()
            .ok_or_else(|| {
                Status::unauthenticated("You are not authenticated. Please provide an identity.")
            })?;

        // verify signature
        let message = get_message(b"create-session", &request, challenge.clone())?;
        keys_lock.verify_signature(pubkey_hash, &message[..], request.metadata())?;

        let (token, expiry) = {
            let time = SystemTime::now();
            let expiry = time
                .checked_add(Duration::from_secs(self.session_expiry))
                .unwrap_or(time);
            (self.new_challenge(), expiry)
        };
        sessions.insert(
            token.clone(),
            Session {
                pubkey: pubkey_hash.to_string(),
                user_ip,
                expiry,
                client_info: request.into_inner(),
            },
        );
        Ok(SessionInfo {
            token: token.to_vec(),
            expiry_time: self.session_expiry * 1000,
        })
    }
}

pub struct SessionGrpcService {
    sess_manager: Arc<SessionManager>,
}

impl SessionGrpcService {
    pub fn new(sess_manager: Arc<SessionManager>) -> Self {
        Self { sess_manager }
    }
}

#[tonic::async_trait]
impl session_proto::session_service_server::SessionService for SessionGrpcService {
    async fn get_challenge(
        &self,
        _request: Request<session_proto::Empty>,
    ) -> Result<Response<session_proto::ChallengeResponse>, Status> {
        let challenge = self.sess_manager.new_challenge();
        Ok(Response::new(session_proto::ChallengeResponse {
            value: challenge.into(),
        }))
    }

    async fn create_session(
        &self,
        request: Request<ClientInfo>,
    ) -> Result<Response<session_proto::SessionInfo>, Status> {
        let session = self.sess_manager.create_session(request)?;
        Ok(Response::new(session))
    }
}
