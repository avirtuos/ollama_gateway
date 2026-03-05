use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use tower::Service;
use uuid::Uuid;

/// Extension carrying the per-TCP-connection UUID.
#[derive(Clone, Debug)]
pub struct ConnectionId(pub String);

/// Tower layer that generates a fresh UUID for every new TCP connection and
/// injects it into each request's extensions.
#[derive(Clone)]
pub struct ConnectionIdLayer;

impl<S> tower::Layer<S> for ConnectionIdLayer {
    type Service = ConnectionIdMakeService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ConnectionIdMakeService { inner }
    }
}

#[derive(Clone)]
pub struct ConnectionIdMakeService<S> {
    inner: S,
}

impl<S, T> Service<T> for ConnectionIdMakeService<S>
where
    S: Service<T>,
    S::Future: Send + 'static,
    S::Response: Send + 'static,
    S::Error: Send + 'static,
{
    type Response = ConnectionIdService<S::Response>;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: T) -> Self::Future {
        let conn_id = Uuid::new_v4().to_string();
        let fut = self.inner.call(req);
        Box::pin(async move {
            let svc = fut.await?;
            Ok(ConnectionIdService { inner: svc, conn_id })
        })
    }
}

#[derive(Clone)]
pub struct ConnectionIdService<S> {
    inner: S,
    conn_id: String,
}

impl<S, ReqBody> Service<axum::http::Request<ReqBody>> for ConnectionIdService<S>
where
    S: Service<axum::http::Request<ReqBody>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: axum::http::Request<ReqBody>) -> Self::Future {
        req.extensions_mut().insert(ConnectionId(self.conn_id.clone()));
        self.inner.call(req)
    }
}
