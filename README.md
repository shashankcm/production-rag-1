# production-rag-1

production RAG application

# API request flow

Client Request ->
Rate Limiter ->
Security Middleware (Injection check, PII masking) ->
Cache Layer (Hit? Returned Cache, Miss? Continue...) ->
Output Validator -> Primary Modal, Retry on failure, Fallback Modal
Metrics and Logs ->
JSON Response
