import sqlite3
import json
import os
import sys
import time
import random
from datetime import datetime
from typing import List, Dict, Optional, Any

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.keyword_summarizer import KeywordSummarizer
from core.text_preprocessor import TextProcessor

class SQLiteChatStorage:
    """
    SQLite-backed chat storage.
    - Stores chat sessions and messages.
    - Messages have: id, session_id, role ('user'|'assistant'), content, metadata (JSON), created_at.
    """

    def __init__(self):
        self.keyword_summarizer = KeywordSummarizer()
        self.tp = TextProcessor()
        self.db_path = os.path.join(project_root, "data", "database", "rag_sqlite.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """
        )
        self.conn.commit()

    def time_random_id(self):
        """Generate a unique session ID based on timestamp and random number."""
        timestamp = int(time.time() * 1000)
        rand = random.randint(1000, 9999)
        return f"session_{timestamp}_{rand}"
    
    # ---------- Session methods ----------
    def create_session(self, session_id: str, name: str) -> str:
        """
        Create a new session if it does not already exist.

        Args:
            session_id: Unique session identifier.
            name: Initial name of the session.

        Returns:
            The session ID.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO sessions (id, name) VALUES (?, ?)",
                (session_id, name),
            )
            self.conn.commit()
        return session_id

    def update_session_name(self, session_id: str, new_name: str) -> bool:
        """
        Generate and update a short, meaningful name for a session.

        This method uses the `KeywordSummarizer` to extract a concise keyword
        or phrase from the provided session content, making it suitable for
        display in the user interface (e.g., session titles or summaries).

        The extracted name is stored in the database, replacing the previous
        session name associated with the given session ID.

        Args:
            session_id (str): Unique identifier of the session to update.
            new_name (str): The text or message content to derive the new name from.

        Returns:
            bool: True if the session name was successfully updated, False otherwise.
        """
        new_name = self.keyword_summarizer.summarize_text(str(new_name), 5).lower()
        cur = self.conn.cursor()
        cur.execute("UPDATE sessions SET name = ? WHERE id = ?", (new_name, session_id))
        self.conn.commit()
        return cur.rowcount > 0 


    def list_sessions(self) -> List[Dict[str, Any]]:
        """Return all sessions ordered by creation date (most recent first)."""

        cur = self.conn.cursor()
        cur.execute("SELECT id, name, created_at FROM sessions ORDER BY created_at DESC")
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session  by ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT id, name, created_at FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def delete_session(self, session_id: str):
        """Delete a session and all its messages."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cur.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self.conn.commit()

    def list_empty_sessions(self) -> List[Dict[str, Any]]:
        """
        Return all sessions that have no messages.

        Returns:
            List of session dicts with keys: id, name, created_at
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT s.id, s.name, s.created_at
            FROM sessions s
            LEFT JOIN messages m ON s.id = m.session_id
            WHERE m.id IS NULL
            ORDER BY s.created_at DESC
        """)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    
    def delete_empty_sessions(self) -> int:
        """
        Delete all sessions that have no messages.

        Returns:
            int: Number of sessions deleted.
        """
        cur = self.conn.cursor()
        cur.execute("""
            DELETE FROM sessions
            WHERE id IN (
                SELECT s.id
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                WHERE m.id IS NULL
            )
        """)
        deleted_count = cur.rowcount
        self.conn.commit()
        return deleted_count

    # ---------- Message methods ----------
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[dict] = None):
        """
        Add a message to a session.

        Args:
            session_id: Session the message belongs to.
            role: 'user' or 'assistant'.
            content: Message text.
            metadata: Optional additional data (sources, etc.)

        Returns:
            Message ID in database.
        """
        metadata_json = json.dumps(metadata) if metadata is not None else None
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO messages (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
            (session_id, role, content, metadata_json),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_user_message(self, session_id: str, content: str, metadata: Optional[dict] = None):
        return self.add_message(session_id, "user", content, metadata)

    def add_ai_message(self, session_id: str, content: str, metadata: Optional[dict] = None):
        return self.add_message(session_id, "assistant", content, metadata)

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Return all messages of a session in chronological order."""

        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, role, content, metadata, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        rows = cur.fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["metadata"] = json.loads(item["metadata"]) if item["metadata"] else None
            out.append(item)
        return out

    def get_last_n_messages(self, session_id: str, n: int = 20) -> List[Dict[str, Any]]:
        """Return the last n messages of a session in chronological order."""

        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, role, content, metadata, created_at FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, n),
        )
        rows = cur.fetchall()

        rows = list(rows)[::-1]
        out = []
        for r in rows:
            item = dict(r)
            item["metadata"] = json.loads(item["metadata"]) if item["metadata"] else None
            out.append(item)
        return out

    def close(self):
        """Close the SQLite connection safely."""
        try:
            self.conn.close()
        except Exception:
            pass

