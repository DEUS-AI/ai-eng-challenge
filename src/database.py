"""SQLite database setup, Customer model, seed data, and lookup functions."""

import re
import sqlite3
from pathlib import Path

from pydantic import BaseModel


class Customer(BaseModel):
    id: int | None = None
    name: str
    phone: str
    iban: str
    secret_question: str
    secret_answer: str
    premium: bool


DB_PATH = Path(__file__).parent.parent / "customers.db"

SEED_CUSTOMERS: list[dict] = [
    {
        "name": "Lisa",
        "phone": "+1122334455",
        "iban": "DE89370400440532013000",
        "secret_question": "Which is the name of my dog?",
        "secret_answer": "Yoda",
        "premium": True,
    },
    {
        "name": "Max Mueller",
        "phone": "+4915112345678",
        "iban": "DE27100777770209299700",
        "secret_question": "What city was I born in?",
        "secret_answer": "Berlin",
        "premium": True,
    },
    {
        "name": "Anna Schmidt",
        "phone": "+4917612345678",
        "iban": "DE44500105175407324931",
        "secret_question": "What is my favorite color?",
        "secret_answer": "Blue",
        "premium": False,
    },
    {
        "name": "John Smith",
        "phone": "+11234567890",
        "iban": "GB29NWBK60161331926819",
        "secret_question": "What is my mother's maiden name?",
        "secret_answer": "Parker",
        "premium": False,
    },
    {
        "name": "Sofia Garcia",
        "phone": "+34612345678",
        "iban": "ES9121000418450200051332",
        "secret_question": "What was the name of my first pet?",
        "secret_answer": "Luna",
        "premium": True,
    },
    {
        "name": "Pierre Dubois",
        "phone": "+33612345678",
        "iban": "FR7630006000011234567890189",
        "secret_question": "What is my favorite movie?",
        "secret_answer": "Amelie",
        "premium": False,
    },
    {
        "name": "Yuki Tanaka",
        "phone": "+81312345678",
        "iban": "DE62370400440532013001",
        "secret_question": "What street did I grow up on?",
        "secret_answer": "Sakura",
        "premium": False,
    },
    {
        "name": "Omar Hassan",
        "phone": "+20101234567",
        "iban": "DE89370400440532013002",
        "secret_question": "What is my lucky number?",
        "secret_answer": "7",
        "premium": True,
    },
]


def _normalize_phone(phone: str) -> str:
    return re.sub(r"[^\d]", "", phone)


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _normalize_iban(iban: str) -> str:
    return iban.replace(" ", "").upper()


def init_db(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT NOT NULL,
            iban TEXT NOT NULL UNIQUE,
            secret_question TEXT NOT NULL,
            secret_answer TEXT NOT NULL,
            premium BOOLEAN NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def seed_db(conn: sqlite3.Connection) -> None:
    count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    if count > 0:
        return
    conn.executemany(
        """INSERT INTO customers (name, phone, iban, secret_question, secret_answer, premium)
           VALUES (:name, :phone, :iban, :secret_question, :secret_answer, :premium)""",
        SEED_CUSTOMERS,
    )
    conn.commit()


def _row_to_customer(row: sqlite3.Row) -> Customer:
    return Customer(
        id=row["id"],
        name=row["name"],
        phone=row["phone"],
        iban=row["iban"],
        secret_question=row["secret_question"],
        secret_answer=row["secret_answer"],
        premium=bool(row["premium"]),
    )


def find_customer_by_identifiers(
    conn: sqlite3.Connection,
    name: str | None = None,
    phone: str | None = None,
    iban: str | None = None,
) -> Customer | None:
    """Find a customer matching at least 2 out of 3 identifiers.

    Returns the first matching customer, or None if fewer than 2 fields match.
    """
    rows = conn.execute("SELECT * FROM customers").fetchall()

    for row in rows:
        matches = 0
        if name and _normalize_name(name) == _normalize_name(row["name"]):
            matches += 1
        if phone and _normalize_phone(phone) == _normalize_phone(row["phone"]):
            matches += 1
        if iban and _normalize_iban(iban) == _normalize_iban(row["iban"]):
            matches += 1

        if matches >= 2:
            return _row_to_customer(row)

    return None


def get_customer_by_iban(conn: sqlite3.Connection, iban: str) -> Customer | None:
    row = conn.execute(
        "SELECT * FROM customers WHERE iban = ?", (_normalize_iban(iban),)
    ).fetchone()
    if not row:
        return None
    return _row_to_customer(row)
