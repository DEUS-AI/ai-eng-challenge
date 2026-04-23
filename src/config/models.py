from enum import Enum
from typing import Optional
from pydantic import BaseModel


class Skill(str, Enum):
    INSURANCE = "insurance"
    INVESTMENTS = "investments"
    ACCOUNTS = "accounts"


SKILL_METADATA = {
    Skill.INSURANCE: {
        "description": "Handles all insurance-related topics including policies, claims, and coverage.",
        "related_requests": ["insurance policy", "claim", "coverage", "premium", "renewal", "accident", "damage"],
    },
    Skill.INVESTMENTS: {
        "description": "Handles investment products, portfolio management, and financial planning.",
        "related_requests": ["investment", "portfolio", "funds", "stocks", "bonds", "retirement", "savings plan"],
    },
    Skill.ACCOUNTS: {
        "description": "Handles account management including balances, transactions, and account settings.",
        "related_requests": ["account", "balance", "transaction", "transfer", "statement", "limit", "card"],
    },
}


class SkillRequest(BaseModel):
    skill: Skill


# ── Output parser models ──────────────────────────────────────────────────────

class IdentityResult(BaseModel):
    """Output of the identity verification step."""
    verified: bool
    secret_question: str = ""
    matched_nif: str = ""


class AccountField(str, Enum):
    BALANCE = "balance"
    IBAN = "iban"
    ACCOUNT_NUMBER = "account_number"
    ACCOUNT_TYPE = "account_type"
    ALL = "all"


class AccountSummary(BaseModel):
    """Structured representation of a customer account returned by get_account_summary.

    Fields:
    - account_number: the unique account identifier (e.g. ACC-001).
    - iban: the IBAN for bank transfers and identification.
    - account_type: 'Premium' for premium customers, 'Regular' otherwise.
    - balance: current account balance in EUR (e.g. 1250.75).
    """
    account_number: str
    iban: str
    account_type: str
    balance: float


class AccountResult(BaseModel):
    """Output of the account type lookup step."""
    account_type: str
    found: bool


class SpecialistDecision(BaseModel):
    """Structured routing decision from the specialist LLM."""
    in_scope: bool
    skill: Optional[Skill] = None
    refusal_message: str = ""
