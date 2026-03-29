"""Tests for agent output schemas and basic logic."""

from src.agents.bouncer import BouncerOutput, MAX_SECRET_ATTEMPTS
from src.agents.greeter import GreeterOutput
from src.agents.specialist import DEPARTMENTS, SUPPORT_NUMBERS, SpecialistOutput


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

    def test_departments_defined(self) -> None:
        assert "loans" in DEPARTMENTS
        assert "cards" in DEPARTMENTS
        assert "insurance" in DEPARTMENTS
        assert "general" in DEPARTMENTS
