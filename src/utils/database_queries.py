from pathlib import Path
import json
import aiofiles
from datetime import datetime

from config.logger import get_logger

logger = get_logger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"

user_accounts_path = str(_DATA_DIR / "users.json")
accounts_type_path = str(_DATA_DIR / "accounts.json")
employees_path = str(_DATA_DIR / "employees.json")
complaints_path = str(_DATA_DIR / "complaints.json")


async def get_user_by_nif(nif: str):
    """Return the full user record for the given NIF, or None if no match."""
    async with aiofiles.open(user_accounts_path, "r") as f:
        content = await f.read()
    user_accounts = json.loads(content)
    for user in user_accounts:
        if user["nif"] == nif:
            return user
    logger.warning("User not found — NIF: %s", nif)
    return None


async def get_user_by_details(name: str, phone: str, iban: str):
    """Return user if at least 2 of 3 details (name, phone, iban) match."""
    async with aiofiles.open(user_accounts_path, "r") as f:
        content = await f.read()
    users = json.loads(content)
    for user in users:
        matches = sum([
            bool(name) and name.lower().strip() == user["name"].lower().strip(),
            bool(phone) and phone.strip() == user["phone"].strip(),
            bool(iban) and iban.strip() == user["iban"].strip(),
        ])
        if matches >= 2:
            return user
    logger.warning("User not found by details — name=%r phone=%r iban=%r", name[:20] if name else None, phone, iban)
    return None


async def verify_secret(nif: str, answer: str) -> bool:
    """Verify the secret answer for a user identified by NIF."""
    async with aiofiles.open(user_accounts_path, "r") as f:
        content = await f.read()
    users = json.loads(content)
    for user in users:
        if user["nif"] == nif:
            return answer.lower().strip() == user["answer"].lower().strip()
    return False


async def get_account_type(nif: str):
    """Return 'premium' or 'regular' for the account associated with the NIF, or None if not registered."""
    async with aiofiles.open(accounts_type_path, "r") as f:
        content = await f.read()
    accounts = json.loads(content)
    for account in accounts:
        if account["nif"] == nif:
            if account["premium"]:
                return 'premium'
            else :
                return 'regular'
    logger.warning("Account not found — NIF: %s", nif)
    return None


async def get_personalized_employee(skill: str):
    """Return the first employee whose skill set covers the requested skill, or None if no one is available."""
    async with aiofiles.open(employees_path, "r") as f:
        content = await f.read()
    employees = json.loads(content)
    for employee in employees:
        if skill in employee["skills"]:
            return employee
    return None


async def get_account_by_nif(nif: str):
    """Fetch account record (account_number, iban, premium) by NIF."""
    async with aiofiles.open(accounts_type_path, "r") as f:
        content = await f.read()
    accounts = json.loads(content)
    for account in accounts:
        if account["nif"] == nif:
            return account
    return None


async def get_all_employees():
    """Return every employee with their skills."""
    async with aiofiles.open(employees_path, "r") as f:
        content = await f.read()
    return json.loads(content)


async def append_complaint(nif: str, complaint: str) -> None:
    """Append a timestamped complaint record to complaints.json, creating the file if it does not exist."""
    try:
        async with aiofiles.open(complaints_path, "r") as f:
            content = await f.read()
        complaints = json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError):
        complaints = []
    complaints.append({
        "nif": nif,
        "complaint": complaint,
        "timestamp": datetime.now().isoformat(),
    })
    try:
        async with aiofiles.open(complaints_path, "w") as f:
            await f.write(json.dumps(complaints, indent=2))
        logger.info("Complaint logged — NIF: %s", nif)
    except Exception:
        logger.exception("Failed to write complaint — NIF: %s", nif)
        raise