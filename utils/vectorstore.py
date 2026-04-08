# Global state mapping mapping for MVP
# In scalable setup, this would be Redis and users would map via session_id
global_store = {
    "vectorstore": None,
    "skills": [],
    "roles": []
}

def set_vectorstore(vs):
    global_store["vectorstore"] = vs

def get_vectorstore():
    return global_store.get("vectorstore")

def set_skills(skills):
    global_store["skills"] = skills

def get_skills():
    return global_store.get("skills", [])

def set_roles(roles):
    global_store["roles"] = roles
    
def get_roles():
    return global_store.get("roles", [])
