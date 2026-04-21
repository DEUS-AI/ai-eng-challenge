import json
import asyncio
from langchain.tools import tool
from utils.database_queries import (
    get_user_by_details,
    get_account_type,
    get_personalized_employee,
    get_account_by_nif,
    get_all_employees,
    append_complaint,
)
from config.models import Skill


@tool 
def verify_account_type(nif:str) -> str:
    """Verify if the customer is a premium, regular customer or not based on NIF. 
    Return costumer type
    """
    account_type = asyncio.run(get_account_type(nif))
    if account_type:
        return account_type
    
    return "Account not found with the provided NIF. Please check the NIF and try again."

@tool
def verify_identity(name: str = "", phone: str = "", iban: str = "") -> str:
    """Verify customer identity by matching at least 2 out of 3 details: name, phone, IBAN.
    Leave empty string for any detail not provided by the customer.
    Returns a secret question if at least 2 details match, otherwise returns no_match.
    """
    user = asyncio.run(get_user_by_details(name, phone, iban))
    if user:
        return json.dumps({"status": "match", "secret_question": user["secret"], "nif": user["nif"]})
    return json.dumps({"status": "no_match"})

@tool
def delegate_hitl(skill: Skill) -> str:
    """Delegate to human agent based on the required skill.

    Available skills: insurance, investments, accounts.
    """
    employee = asyncio.run(get_personalized_employee(skill.value))
    if employee:
        return employee['name']
    return "no_employee_available"


@tool
def get_account_summary(nif: str) -> str:
    """Return a summary of the authenticated customer's account: account number, IBAN and account type.
    Use whenever the customer asks to see their account details.
    """
    account = asyncio.run(get_account_by_nif(nif))
    if account:
        account_type = "Premium" if account.get("premium") else "Regular"
        return json.dumps({
            "account_number": account["account_number"],
            "iban": account["iban"],
            "account_type": account_type,
        })
    return json.dumps({"error": "Account not found."})


@tool
def get_available_specialists() -> str:
    """Return the list of available specialist employees and the topics each one handles.
    Use when the customer asks who can help them or which specialist covers a given topic.
    """
    employees = asyncio.run(get_all_employees())
    result = [{"name": e["name"], "skills": e["skills"]} for e in employees]
    return json.dumps(result)


@tool
def log_complaint(nif: str, complaint: str) -> str:
    """Register a customer complaint. Summarise the complaint in one sentence before calling.
    Use when the customer explicitly wants to file a complaint or report a problem.
    """
    asyncio.run(append_complaint(nif, complaint))
    return "Your complaint has been registered. A specialist will review it shortly."

