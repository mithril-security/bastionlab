use std::collections::HashSet;
use std::net::SocketAddr;
use std::time::{Duration, Instant, SystemTime};

use bytes::Bytes;
use prost::Message;
use tonic::metadata::KeyRef;
use tonic::{Request, Response, Status};

use crate::auth::KeyManagement;
use crate::session_proto::{ClientInfo, Empty, SessionInfo};
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
        .ok_or_else(|| Status::invalid_argument("No access token in request metadata"))?;
    Ok(Some(meta.to_bytes().map_err(|_| {
        Status::invalid_argument("Could not decode accesstoken")
    })?))
}

#[derive(Debug, Clone)]
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
    ban_time: u64,
    challenges: Mutex<HashSet<[u8; 32]>>,
    blocklist: Arc<RwLock<HashMap<String, Instant>>>,
}

impl SessionManager {
    pub fn new(keys: Option<KeyManagement>, session_expiry: u64, ban_time: u64) -> Self {
        Self {
            keys: keys.map(Mutex::new),
            sessions: Default::default(),
            session_expiry,
            ban_time,
            challenges: Default::default(),
            blocklist: Default::default(),
        }
    }

    pub fn auth_enabled(&self) -> bool {
        self.keys.is_some()
    }

    pub fn get_user_id(&self, token: Option<Bytes>) -> Result<String, Status> {
        let token_bytes = match &token {
            Some(v) => &v[..],
            None => &[0u8; 32],
        };
        let sessions = self.sessions.read().unwrap();
        let session = sessions
            .get(token_bytes)
            .ok_or(Status::aborted("Session not found!"))?;

        let user_id = session.pubkey.clone();
        Ok(user_id)
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
        let mut sessions = self.sessions.write().unwrap();
        let keys_lock = self.keys.as_ref().map(|l| l.lock().expect("Poisoned lock"));
        let end = "-bin";
        let pat = "signature-";

        let user_ip = match request.remote_addr() {
            Some(user_ip) => user_ip,
            _ => return Err(Status::aborted("Could not fetch IP Address from request")),
        };

        match self.auth_enabled() {
            true => {
                let challenge = self.check_challenge(&request)?;

                for key in request.metadata().keys() {
                    match key {
                        KeyRef::Binary(key) => {
                            let key_string = key.to_string();
                            let key = key_string.strip_suffix(end).unwrap();
                            if key.contains(pat) {
                                if let Some(key) = key.split(pat).last() {
                                    let key = key.to_string();
                                    {
                                        let mut blocklist = self.blocklist.write().unwrap();
                                        if let Some(blocked) = blocklist.get(&key) {
                                            if blocked.elapsed().as_secs() < self.ban_time {
                                                return Err(Status::aborted(
                                                "This user is temporarily blocked due to suspicious behaviour. Please try again later.",
                                            ));
                                            } else {
                                                blocklist.remove(&key);
                                            }
                                        }
                                    }
                                    if let Some(ref keys) = keys_lock {
                                        let lock = keys;
                                        let message = get_message(
                                            b"create-session",
                                            &request,
                                            challenge.clone(),
                                        )?;
                                        //Existing sessions are deleted before a new session is created
                                        for (token, session) in sessions.clone().iter() {
                                            if session.pubkey == key {
                                                sessions.remove(token.as_ref());
                                            }
                                        }
                                        lock.verify_signature(
                                            &key,
                                            &message[..],
                                            request.metadata(),
                                        )?;
                                    }
                                    let (token, expiry) = {
                                        let expiry = match SystemTime::now()
                                            .checked_add(Duration::from_secs(self.session_expiry))
                                        {
                                            Some(v) => v,
                                            None => SystemTime::now(),
                                        };
                                        (self.new_challenge(), expiry)
                                    };
                                    sessions.insert(
                                        token.clone(),
                                        Session {
                                            pubkey: key,
                                            user_ip,
                                            expiry,
                                            client_info: request.into_inner(),
                                        },
                                    );
                                    return Ok(SessionInfo {
                                        token: token.to_vec(),
                                        expiry_time: self.session_expiry * 1000,
                                    });
                                } else {
                                    return Err(Status::aborted(
                                        "User signing key not found in request!",
                                    ));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                return Err(Status::aborted("Signature header missing or malformed!"));
            }
            false => {
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
        };
    }

    pub fn delete_session(&self, token: Bytes) -> () {
        let mut tokens = self.sessions.write().unwrap();
        tokens.remove(token.as_ref());
    }

    pub fn block_user(&self, user_id: String, timestamp: Instant) {
        let mut blocklist = self.blocklist.write().unwrap();
        blocklist.insert(user_id, timestamp);
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

    async fn delete_session(
        &self,
        request: Request<session_proto::Empty>,
    ) -> Result<Response<session_proto::Empty>, Status> {
        if self.sess_manager.auth_enabled() {
            let token = get_token(&request, true)?;
            self.sess_manager.delete_session(token.unwrap());
        }
        Ok(Response::new(Empty {}))
    }
}
