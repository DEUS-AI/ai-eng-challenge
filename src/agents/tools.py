import asyncio
from langchain.tools import tool
from utils.database_queries import get_user_by_nif, get_account_type, get_personalized_employee
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
def verify_identity(nif: str) -> str:
    """Verify customer identity using provided information."""
    user = asyncio.run(get_user_by_nif(nif))
    if user:
        return "Identity verified successfully"
    
    return "Identity verification failed"

@tool
def delegate_hitl(skill: Skill) -> str:
    """Delegate to human agent based on the required skill.
    
    Available skills: insurance, investments, accounts.
    """
    employee = asyncio.run(get_personalized_employee(skill.value))
    if employee:
        return employee['name']
    
    return "no_employee_available"

