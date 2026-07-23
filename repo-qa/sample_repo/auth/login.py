"""Authentication module: handles user login and session tokens."""

import hashlib
import secrets
from datetime import datetime, timedelta

SESSION_STORE = {}


def hash_password(password: str, salt: str) -> str:
    """Hash a password with a salt using SHA-256."""
    return hashlib.sha256((password + salt).encode()).hexdigest()


def create_session_token(user_id: str) -> str:
    """Generate a random session token and store it with an expiry."""
    token = secrets.token_hex(32)
    SESSION_STORE[token] = {
        "user_id": user_id,
        "expires_at": datetime.utcnow() + timedelta(hours=24),
    }
    return token


def login(username: str, password: str, db_lookup_fn) -> str:
    """
    Validate credentials against the database and return a session token.
    Raises ValueError if credentials are invalid.
    """
    user = db_lookup_fn(username)
    if user is None:
        raise ValueError("User not found")

    hashed = hash_password(password, user["salt"])
    if hashed != user["password_hash"]:
        raise ValueError("Invalid password")

    return create_session_token(user["id"])


def validate_session(token: str) -> bool:
    """Check whether a session token is present and not expired."""
    session = SESSION_STORE.get(token)
    if session is None:
        return False
    return datetime.utcnow() < session["expires_at"]


def logout(token: str) -> None:
    """Invalidate a session token."""
    SESSION_STORE.pop(token, None)
