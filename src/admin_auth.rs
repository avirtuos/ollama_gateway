//! Password hashing and session-token helpers for the admin UI.
//!
//! The admin password is stored as an Argon2id PHC string (salt embedded); the
//! plaintext is never written to disk. Sessions live in memory only, so a
//! gateway restart requires a fresh login.

use std::collections::HashSet;
use std::sync::Arc;

use argon2::{
    password_hash::{rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use tokio::sync::RwLock;

/// Name of the cookie that carries the admin session token.
pub const SESSION_COOKIE_NAME: &str = "admin_session";

/// Hash a plaintext password into an Argon2id PHC string.
///
/// CPU-bound (tens of milliseconds) — call from `spawn_blocking` on the request path.
pub fn hash_password(password: &str) -> anyhow::Result<String> {
    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| anyhow::anyhow!("failed to hash admin password: {e}"))
}

/// Verify a plaintext password against a stored PHC string.
///
/// Returns false for a malformed or empty hash rather than erroring, so a
/// corrupt config can never be read as "any password accepted".
pub fn verify_password(phc: &str, password: &str) -> bool {
    match PasswordHash::new(phc) {
        Ok(parsed) => Argon2::default()
            .verify_password(password.as_bytes(), &parsed)
            .is_ok(),
        Err(_) => false,
    }
}

/// 256 bits of CSPRNG-backed session token (two UUIDv4s, hyphens stripped).
pub fn new_session_token() -> String {
    format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    )
}

/// Extract the session token from a `Cookie` header value, if present.
pub fn session_token_from_cookie_header(header: &str) -> Option<String> {
    header.split(';').find_map(|pair| {
        let (name, value) = pair.trim().split_once('=')?;
        (name == SESSION_COOKIE_NAME).then(|| value.to_string())
    })
}

/// In-memory store of valid admin session tokens. Not persisted — a gateway
/// restart invalidates every session and requires a fresh login.
#[derive(Clone, Default)]
pub struct SessionStore(Arc<RwLock<HashSet<String>>>);

impl SessionStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mint and remember a new session token.
    pub async fn create(&self) -> String {
        let token = new_session_token();
        self.0.write().await.insert(token.clone());
        token
    }

    pub async fn is_valid(&self, token: &str) -> bool {
        self.0.read().await.contains(token)
    }

    pub async fn invalidate(&self, token: &str) {
        self.0.write().await.remove(token);
    }

    /// Drop every outstanding session, forcing all clients to re-login.
    pub async fn invalidate_all(&self) {
        self.0.write().await.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_roundtrip() {
        let phc = hash_password("correct horse").unwrap();
        assert!(phc.starts_with("$argon2id$"));
        assert!(verify_password(&phc, "correct horse"));
        assert!(!verify_password(&phc, "wrong horse"));
        assert!(!verify_password(&phc, ""));
    }

    #[test]
    fn same_password_hashes_differently() {
        let a = hash_password("pw").unwrap();
        let b = hash_password("pw").unwrap();
        assert_ne!(a, b, "salt must be random per hash");
        assert!(verify_password(&a, "pw") && verify_password(&b, "pw"));
    }

    #[test]
    fn malformed_hash_rejects_everything() {
        assert!(!verify_password("", "pw"));
        assert!(!verify_password("not-a-phc-string", "pw"));
        assert!(!verify_password("$argon2id$garbage", ""));
    }

    #[test]
    fn session_tokens_are_unique_and_long() {
        let a = new_session_token();
        let b = new_session_token();
        assert_eq!(a.len(), 64);
        assert_ne!(a, b);
    }
}
