"""Database connection and user lookup utilities."""

import sqlite3
from contextlib import contextmanager

DB_PATH = "app.db"


@contextmanager
def get_connection():
    """Open a SQLite connection and ensure it closes cleanly."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_schema():
    """Create the users table if it doesn't already exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def find_user_by_username(username: str):
    """Look up a single user row by username, or None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None


def insert_user(user_id: str, username: str, password_hash: str, salt: str, created_at: str):
    """Insert a new user row into the users table."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO users (id, username, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, password_hash, salt, created_at),
        )
        conn.commit()
