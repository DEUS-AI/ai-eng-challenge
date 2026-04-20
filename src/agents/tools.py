import json
import asyncio
from langchain.tools import tool
from utils.database_queries import get_user_by_details, get_account_type, get_personalized_employee
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
        return f" {employee['name']} is available for further assistance."
    
    return "No available employee found with the required skill. Request has been escalated to a human agent for further assistance."

