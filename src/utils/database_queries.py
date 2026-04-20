

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