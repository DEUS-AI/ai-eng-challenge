"""Tests for agent output schemas and basic logic."""

from src.agents.bouncer import BouncerOutput, MAX_SECRET_ATTEMPTS, _normalize_answer
from src.agents.greeter import GreeterOutput
from src.agents.specialist import SUPPORT_NUMBERS, SpecialistOutput


class TestGreeterOutput:
    def test_default_values(self) -> None:
        output = GreeterOutput()
        assert output.name is None
        assert output.phone is None
        assert output.iban is None
        assert output.all_collected is False
        assert output.message == ""

    def test_partial_collection(self) -> None:
        output = GreeterOutput(name="Lisa", message="Got your name!")
        assert output.name == "Lisa"
        assert output.phone is None
        assert output.all_collected is False

    def test_full_collection(self) -> None:
        output = GreeterOutput(
            name="Lisa",
            phone="+1122334455",
            iban="DE89370400440532013000",
            all_collected=True,
            message="Thank you, I have all your details.",
        )
        assert output.all_collected is True


class TestBouncerOutput:
    def test_default_values(self) -> None:
        output = BouncerOutput()
        assert output.verified is False
        assert output.tier is None
        assert output.secret_correct is None

    def test_verified_premium(self) -> None:
        output = BouncerOutput(verified=True, tier="premium", secret_correct=True)
        assert output.verified is True
        assert output.tier == "premium"

    def test_max_attempts_constant(self) -> None:
        assert MAX_SECRET_ATTEMPTS == 3


class TestSpecialistOutput:
    def test_default_department(self) -> None:
        output = SpecialistOutput()
        assert output.department == "general"

    def test_support_numbers(self) -> None:
        assert SUPPORT_NUMBERS["premium"] == "+1999888999"
        assert SUPPORT_NUMBERS["regular"] == "+1112112112"

    def test_department_literal(self) -> None:
        output = SpecialistOutput(department="loans")
        assert output.department == "loans"


class TestBouncerSecretNormalization:
    """Test deterministic secret answer comparison."""

    def test_case_insensitive(self) -> None:
        assert _normalize_answer("Yoda") == _normalize_answer("yoda")

    def test_strips_whitespace(self) -> None:
        assert _normalize_answer("  Yoda  ") == _normalize_answer("Yoda")

    def test_exact_match(self) -> None:
        assert _normalize_answer("Blue") == _normalize_answer("Blue")

    def test_mismatch(self) -> None:
        assert _normalize_answer("Wrong") != _normalize_answer("Yoda")
