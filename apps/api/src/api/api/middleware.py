from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import uuid
import logging

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that adds a unique request ID to each incoming request and logs the request details."""

    async def dispatch(self, request: Request, call_next):

        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        logger.info(
            f"Received request: {request.method} {request.url} - Request ID: {request_id}")

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id

        logger.info(
            f"Completed request: {request.method} {request.url} - Request ID: {request_id} - Status Code: {response.status_code}")

        return response
