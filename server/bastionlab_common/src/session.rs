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

fn verify_ip(stored: &SocketAddr, recv: &SocketAddr) -> bool {
    stored.ip().eq(&recv.ip())
}

pub fn get_token<T>(req: &Request<T>, auth_enabled: bool) -> Result<Option<Bytes>, Status> {
    if !auth_enabled {
        return Ok(None);
    }
    let meta = req
        .metadata()
        .get_bin("accesstoken-bin")
        .ok_or_else(|| Status::invalid_argument("No accesstoken in request metadata"))?;
    Ok(Some(meta.to_bytes().map_err(|_| {
        Status::invalid_argument("Could not decode accesstoken")
    })?))
}

#[derive(Debug)]
pub struct Session {
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

    pub fn verify_request<T>(&self, req: &Request<T>) -> Result<Option<Bytes>, Status> {
        let token = get_token(&req, self.auth_enabled())?;
        let lock = self.keys.as_ref().map(|l| l.lock().expect("Poisoned lock"));
        match lock {
            Some(_) => match token.clone() {
                Some(token) => {
                    let mut tokens = self.sessions.write().unwrap();
                    let session = tokens
                        .get(token.as_ref())
                        .ok_or(Status::aborted("Session not found!"))?;
                    let recv_ip = &req
                        .remote_addr()
                        .ok_or(Status::aborted("User IP unavailable"))?;
                    let curr_time = SystemTime::now();

                    if !verify_ip(&session.user_ip, &recv_ip) {
                        return Err(Status::aborted("Unknown IP Address!"));
                    }

                    if curr_time.gt(&session.expiry) {
                        tokens.remove(token.as_ref());
                        return Err(Status::aborted("Session Expired"));
                    }
                }

                None => {
                    return Err(Status::aborted("Session not found!"));
                }
            },

            None => drop(lock),
        }

        Ok(token)
    }

    pub fn get_client_info(&self, token: Option<Bytes>) -> Result<ClientInfo, Status> {
        let sessions = self.sessions.write().unwrap();
        let token = match &token {
            Some(v) => &v[..],
            None => &[0u8; 32],
        };
        let session = sessions
            .get(token)
            .ok_or(Status::aborted("Session not found!"))?;
        Ok(session.client_info.clone())
    }

    fn new_challenge(&self) -> [u8; 32] {
        let rng = ring::rand::SystemRandom::new();
        loop {
            if let Ok(challenge) = ring::rand::generate(&rng) {
                let challenge: [u8; 32] = challenge.expose();
                let mut lock = self.challenges.lock().unwrap();
                lock.insert(challenge);
                return challenge;
            }
        }
    }

    fn check_challenge<T: Message>(&self, request: &Request<T>) -> Result<Bytes, Status> {
        let mut lock = self.challenges.lock().unwrap();
        if let Some(challenge) = request.metadata().get_bin("challenge-bin") {
            let challenge_bytes = challenge.to_bytes().map_err(|_| {
                Status::invalid_argument(format!("Could not decode challenge {:?}", challenge))
            })?;
            let challenge = challenge_bytes.as_ref();

            if !lock.remove(challenge) {
                return Err(Status::permission_denied("Challenge not found!"));
            }

            Ok(challenge_bytes)
        } else {
            return Err(Status::permission_denied("No challenge in request!"));
        }
    }

    // TODO: move grpc specific things to the grpc service and not the session manager
    fn create_session(&self, request: Request<ClientInfo>) -> Result<SessionInfo, Status> {
        let challenge = self.check_challenge(&request)?;
        let mut sessions = self.sessions.write().unwrap();
        let keys_lock = self.keys.as_ref().map(|l| l.lock().expect("Poisoned lock"));
        let end = "-bin";
        let pat = "signature-";
        let mut public_key = String::new();
        if let Some(user_ip) = request.remote_addr() {
            for key in request.metadata().keys() {
                match key {
                    KeyRef::Binary(key) => {
                        let key = key.to_string();
                        if let Some(key) = key.strip_suffix(end) {
                            if key.contains(pat) {
                                if let Some(key) = key.split(pat).last() {
                                    if let Some(ref keys) = keys_lock {
                                        let lock = keys;
                                        let message = get_message(
                                            b"create-session",
                                            &request,
                                            challenge.clone(),
                                        )?;
                                        lock.verify_signature(
                                            key,
                                            &message[..],
                                            request.metadata(),
                                        )?;
                                        public_key.push_str(key);
                                    }
                                } else {
                                    return Err(Status::aborted(
                                        "User signing key not found in request!",
                                    ));
                                }
                            }
                        } else {
                            return Err(Status::aborted("User signing key not found in request!"));
                        }
                    }
                    _ => (),
                }
            }
            let (token, expiry) = if !self.auth_enabled() {
                ([0u8; 32], SystemTime::now())
            } else {
                let expiry =
                    match SystemTime::now().checked_add(Duration::from_secs(self.session_expiry)) {
                        Some(v) => v,
                        None => SystemTime::now(),
                    };
                (self.new_challenge(), expiry)
            };

            sessions.insert(
                token.clone(),
                Session {
                    user_ip,
                    expiry,
                    client_info: request.into_inner(),
                },
            );
            Ok(SessionInfo {
                token: token.to_vec(),
            })
        } else {
            Err(Status::aborted("Could not fetch IP Address from request"))
        }
    }

    fn refresh_session<T>(&self, req: &Request<T>) -> Result<(), Status> {
        if let Some(token) = get_token(req, self.auth_enabled())? {
            let mut sessions = self.sessions.write().unwrap();
            let session = sessions
                .get_mut(&token[..])
                .ok_or(Status::aborted("Session not found!"))?;

            let e = session
                .expiry
                .checked_add(Duration::from_secs(self.session_expiry))
                .ok_or(Status::aborted("Malformed session expiry time!"))?;

            session.expiry = e;
        }
        Ok(())
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

    async fn refresh_session(
        &self,
        request: Request<session_proto::Empty>,
    ) -> Result<Response<session_proto::Empty>, Status> {
        self.sess_manager.refresh_session(&request)?;
        Ok(Response::new(session_proto::Empty {}))
    }
}
