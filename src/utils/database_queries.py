

import json
import aiofiles


user_accounts_path = "src/data/users.json"
accounts_type_path = "src/data/accounts.json"
employees_path = "src/data/employees.json"


async def get_user_by_nif(nif: str):
    """Fetch user details based on NIF."""
    async with aiofiles.open(user_accounts_path, "r") as f:
        content = await f.read()
    user_accounts = json.loads(content)
    for user in user_accounts:
        if user["nif"] == nif:
            return user
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


async def get_account_type(nif:str): 

    """Fetch account type based on NIF."""
    async with aiofiles.open(accounts_type_path, "r") as f:
        content = await f.read()
    accounts = json.loads(content)
    for account in accounts:
        if account["nif"] == nif:
            if account["premium"]:
                return 'premium'
            else :
                return 'regular'
    return None


async def get_personalized_employee(skill: str):
    """Fetch employee details based on skill."""
    async with aiofiles.open(employees_path, "r") as f:
        content = await f.read()
    employees = json.loads(content)
    for employee in employees:
        if skill in employee["skills"]:
            return employee
    return None