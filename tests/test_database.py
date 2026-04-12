"""Tests for database matching logic and normalization."""

import sqlite3

import pytest

from src.database import (
    Customer,
    find_customer_by_identifiers,
    init_db,
    seed_db,
)


@pytest.fixture
def db() -> sqlite3.Connection:
    """Create an in-memory database with seed data."""
    conn = init_db(":memory:")
    seed_db(conn)
    return conn


class TestFindCustomerByIdentifiers:
    """Test 2-of-3 matching logic."""

    def test_match_3_of_3(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="Lisa", phone="+1122334455", iban="DE89370400440532013000"
        )
        assert result is not None
        assert result.name == "Lisa"
        assert result.premium is True

    def test_match_name_and_phone(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="Lisa", phone="+1122334455"
        )
        assert result is not None
        assert result.name == "Lisa"

    def test_match_name_and_iban(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="Lisa", iban="DE89370400440532013000"
        )
        assert result is not None
        assert result.name == "Lisa"

    def test_match_phone_and_iban(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, phone="+1122334455", iban="DE89370400440532013000"
        )
        assert result is not None
        assert result.name == "Lisa"

    def test_no_match_1_of_3(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="Lisa", phone="+9999999999", iban="XX00000000000000"
        )
        assert result is None

    def test_no_match_0_of_3(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="Nobody", phone="+0000000000", iban="XX00000000000000"
        )
        assert result is None

    def test_no_match_none_provided(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(db)
        assert result is None

    def test_no_match_only_one_provided(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(db, name="Lisa")
        assert result is None


class TestNameNormalization:
    def test_case_insensitive(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="lisa", phone="+1122334455"
        )
        assert result is not None
        assert result.name == "Lisa"

    def test_uppercase(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="LISA", phone="+1122334455"
        )
        assert result is not None

    def test_extra_whitespace(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="  Lisa  ", phone="+1122334455"
        )
        assert result is not None

    def test_multi_word_name(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="max mueller", phone="+4915112345678"
        )
        assert result is not None
        assert result.name == "Max Mueller"


class TestPhoneNormalization:
    def test_strips_plus(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="Lisa", phone="1122334455"
        )
        assert result is not None

    def test_strips_dashes(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="Lisa", phone="+1-122-334-455"
        )
        assert result is not None

    def test_strips_spaces(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="Lisa", phone="+1 122 334 455"
        )
        assert result is not None

    def test_strips_parens(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="Lisa", phone="(1) 122-334-455"
        )
        assert result is not None


class TestIbanNormalization:
    def test_lowercase_iban(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="Lisa", iban="de89370400440532013000"
        )
        assert result is not None

    def test_iban_with_spaces(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="Lisa", iban="DE89 3704 0044 0532 0130 00"
        )
        assert result is not None



class TestCustomerTier:
    def test_premium_customer(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="Lisa", phone="+1122334455"
        )
        assert result is not None
        assert result.premium is True

    def test_regular_customer(self, db: sqlite3.Connection) -> None:
        result = find_customer_by_identifiers(
            db, name="Anna Schmidt", phone="+4917612345678"
        )
        assert result is not None
        assert result.premium is False
