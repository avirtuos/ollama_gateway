## Ollama Gateway

We need a rust based ollama gateway what acts as a proxy to an ollama instance. This proxy should add two features ontop of ollama. Firstly, ollama lacks any authentication mechanism so I want to add open ai like token authentication using the same mechanism as open ai (which I believe is a header check for the auth token). Second, we want to add langfuse support for tracing LLM calls. We want the langfuse details as well as the auth tokens to be configurable via a text config file whos location is supplied to the binary via a command line argument. The config file should allow definig an "app name" for each auth token and that app name should be added to the langfuse traces.

It might be worth looking into pingora rust crate as the basis for our proxy.
