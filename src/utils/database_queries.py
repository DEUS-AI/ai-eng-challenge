

import json
import aiofiles


user_accounts_path = "src/data/users.json"

async def get_user_by_nif(nif: str):
    """Fetch user details based on NIF."""
    async with aiofiles.open(user_accounts_path, "r") as f:
        content = await f.read()
    user_accounts = json.loads(content)
    for user in user_accounts:
        if user["nif"] == nif:
            return user
    return None