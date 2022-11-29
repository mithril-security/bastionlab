use std::net::SocketAddr;
use std::time::{Duration, SystemTime};

use bytes::Bytes;
use prost::Message;
use tonic::metadata::KeyRef;
use tonic::{Request, Response, Status};

use crate::auth::KeyManagement;
use crate::session_proto::ClientInfo;
use crate::{prelude::*, session_proto};

fn get_message<T: Message>(method: &[u8], req: &Request<T>) -> Result<Vec<u8>, Status> {
    let meta = req
        .metadata()
        .get_bin("challenge-bin")
        .ok_or_else(|| Status::invalid_argument("No challenge in request metadata"))?;
    let challenge = meta
        .to_bytes()
        .map_err(|_| Status::invalid_argument("Could not decode challenge"))?;

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

fn get_token<T>(req: &Request<T>) -> Result<Bytes, Status> {
    let meta = req
        .metadata()
        .get_bin("accesstoken-bin")
        .ok_or_else(|| Status::invalid_argument("No accesstoken in request metadata"))?;
    Ok(meta
        .to_bytes()
        .map_err(|_| Status::invalid_argument("Could not decode accesstoken"))?)
}

#[derive(Debug)]
pub struct Session {
    pub user_ip: SocketAddr,
    pub expiry: SystemTime,
    pub public_key: String,
    pub client_info: ClientInfo,
}

#[derive(Debug)]
pub struct SessionManager {
    keys: Mutex<KeyManagement>,
    sessions: Arc<RwLock<HashMap<[u8; 32], Session>>>,
    session_expiry: u64,
}

impl SessionManager {
    pub fn new(keys: KeyManagement, session_expiry: u64) -> Self {
        Self {
            keys: Mutex::new(keys),
            sessions: Default::default(),
            session_expiry,
        }
    }

    pub fn verify_request<T>(&self, req: &Request<T>) -> Result<(), Status> {
        let remote_addr = &req.remote_addr();
        let token = get_token(req)?;
        let mut tokens = self.sessions.write().unwrap();
        if let Some(recv_ip) = remote_addr {
            if let Some(Session {
                user_ip, expiry, ..
            }) = tokens.get(token.as_ref())
            {
                let curr_time = SystemTime::now();
                if !verify_ip(&user_ip, &recv_ip) {
                    return Err(Status::aborted("Unknown IP Address!"));
                }
                if curr_time.gt(expiry) {
                    tokens.remove(token.as_ref());
                    return Err(Status::aborted("Session Expired"));
                }
            }
        }

        Ok(())
    }

    pub fn get_client_info<T>(&self, req: &Request<T>) -> Result<ClientInfo, Status> {
        let token = get_token(req)?;
        let sessions = self.sessions.write().unwrap();
        let session = sessions
            .get(&token[..])
            .ok_or(Status::aborted("Session not found!"))?;
        Ok(session.client_info.clone())
    }

    fn new_challenge(&self) -> [u8; 32] {
        let rng = ring::rand::SystemRandom::new();
        loop {
            if let Ok(challenge) = ring::rand::generate(&rng) {
                return challenge.expose();
            }
        }
    }

    // TODO: move grpc specific things to the grpc service and not the session manager
    fn create_session(
        &self,
        request: Request<ClientInfo>,
    ) -> Result<session_proto::SessionInfo, Status> {
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
                                    let lock = self.keys.lock().unwrap();
                                    let message = get_message(b"create-session", &request)?;
                                    lock.verify_signature(key, &message[..], request.metadata())?;
                                    public_key.push_str(key);
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
            let mut sessions = self.sessions.write().unwrap();
            let token = self.new_challenge();
            let Some(expiry) = SystemTime::now().checked_add(Duration::from_secs(self.session_expiry)) else {
                return Err(Status::aborted("Could not create expiry for session"));
            };

            sessions.insert(
                token.clone(),
                Session {
                    user_ip,
                    expiry,
                    public_key,
                    client_info: request.into_inner(),
                },
            );

            Ok(session_proto::SessionInfo {
                token: token.to_vec(),
            })
        } else {
            Err(Status::aborted("Could not fetch IP Address from request"))
        }
    }

    fn refresh_session<T>(&self, req: &Request<T>) -> Result<(), Status> {
        let token = get_token(req)?;
        let mut sessions = self.sessions.write().unwrap();
        let session = sessions
            .get_mut(&token[..])
            .ok_or(Status::aborted("Session not found!"))?;

        let e = session
            .expiry
            .checked_add(Duration::from_secs(self.session_expiry))
            .ok_or(Status::aborted("Malformed session expiry time!"))?;

        session.expiry = e;
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
