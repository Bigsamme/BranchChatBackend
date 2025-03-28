from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import jwt
from pydantic import BaseModel
from datetime import datetime
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_authenticated_user(authorization: str = Header(...)) -> str:
    """
    Reads the user ID out of the Bearer token.
    Skips signature verification for brevity; 
    you should validate properly in production.
    """
    token = authorization.split("Bearer ")[-1].strip()
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")
        return user_id
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

def title(text: str) -> str:
    """
    Call the Gemini model to generate a short chat title from the user's message using a streaming response.
    """
    client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))
    response = client.models.generate_content_stream(
        model="gemini-2.0-flash",
        contents=[f"Can you make this into a short title for a chat: {text}\nReturn only the title text and nothing else."]
    )
    full_text = ""
    for chunk in response:
        full_text += chunk.text
    return full_text.strip()

def generate_ai_reply_with_context(chat_id: str, prompt: str) -> str:
    """
    Retrieve the last 100 messages for the chat and send them along with the new prompt.
    """
    # Retrieve conversation history
    msg_resp = supabase.table("messages") \
                       .select("*") \
                       .eq("chat_id", chat_id) \
                       .order("created_at", desc=False) \
                       .limit(100) \
                       .execute()
    messages = msg_resp.data or []

    # Build a "context" prompt from the existing messages
    context_prompt = ""
    for message in messages:
        context_prompt += f"{message['role']}: {message['content']}\n"

    # Append the new prompt as the user's latest message
    # context_prompt += f"user: {prompt}\n"  # This line has been removed

    client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))
    chat_obj = client.chats.create(model="gemini-2.0-flash")
    stream_response = chat_obj.send_message_stream(context_prompt)
    full_text = ""
    for chunk in stream_response:
        full_text += chunk.text
    return full_text.strip()

@app.post("/chats")
def create_empty_chat(user_id: str = Depends(get_authenticated_user)):
    """
    Creates a new chat with a placeholder title, returns { chat_id, title }.
    No AI calls here, so it's fast.
    """
    placeholder_title = "Untitled Chat"
    data = {
        "user_id": user_id,
        "title": placeholder_title,
        "created_at": datetime.utcnow().isoformat()
    }
    chat_response = supabase.table("chats").insert(data).execute()
    if not chat_response.data:
        raise HTTPException(status_code=500, detail="Failed to create empty chat")

    chat_id = chat_response.data[0]["id"]
    return {"chat_id": chat_id, "title": placeholder_title}

@app.get("/chats")
def get_chats(user_id: str = Depends(get_authenticated_user)):
    """
    Return a list of chats, newest first.
    """
    response = supabase.table("chats") \
                       .select("id, title, created_at, user_id, branch_of") \
                       .eq("user_id", user_id) \
                       .order("created_at", desc=True) \
                       .execute()
    for chat in response.data:
        chat["name"] = chat.pop("title")
    return response.data

class MessageRequest(BaseModel):
    content: str

@app.get("/chats/{chat_id}/messages")
def get_chat_messages(chat_id: str, user_id: str = Depends(get_authenticated_user)):
    """
    Return all messages for a specific chat, sorted oldest to newest.
    """
    msg_resp = supabase.table("messages") \
                       .select("*") \
                       .eq("chat_id", chat_id) \
                       .order("created_at", desc=False) \
                       .execute()
    return msg_resp.data

@app.post("/chats/{chat_id}/messages")
def create_message(chat_id: str, req: MessageRequest, user_id: str = Depends(get_authenticated_user)):
    """
    1) Store the user's new message in the DB.
    2) If the chat is still "Untitled Chat," rename it using the user's text.
    3) Generate an AI reply with context, store it, and return both new messages.
    """
    # 1. Insert the user's message
    user_msg_data = {
        "chat_id": chat_id,
        "user_id": user_id,
        "role": "user",
        "content": req.content,
        "created_at": datetime.utcnow().isoformat()
    }
    user_insert = supabase.table("messages").insert(user_msg_data).execute()
    if not user_insert.data:
        raise HTTPException(status_code=500, detail="Failed to insert user message")

    # 2. Possibly rename the chat if it's still a placeholder
    chat_resp = supabase.table("chats").select("*").eq("id", chat_id).execute()
    if not chat_resp.data:
        raise HTTPException(status_code=404, detail="Chat not found")
    current_title = chat_resp.data[0]["title"]
    if current_title == "Untitled Chat":
        new_title = title(req.content)
        supabase.table("chats").update({"title": new_title}).eq("id", chat_id).execute()

    # 3. Stream the AI reply and store it after completion
    def stream_response():
        client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))
        chat_obj = client.chats.create(model="gemini-2.0-flash")

        # Rebuild context from last 100 messages
        msg_resp = supabase.table("messages") \
                           .select("*") \
                           .eq("chat_id", chat_id) \
                           .order("created_at", desc=False) \
                           .limit(100) \
                           .execute()
        messages = msg_resp.data or []
        context_prompt = ""
        for message in messages:
            context_prompt += f"{message['role']}: {message['content']}\n"

        stream = chat_obj.send_message_stream(context_prompt)

        full_text = ""
        for chunk in stream:
            full_text += chunk.text
            yield chunk.text

        # Store assistant message after streaming is complete
        ai_msg_data = {
            "chat_id": chat_id,
            "user_id": user_id,
            "role": "assistant",
            "content": full_text,
            "created_at": datetime.utcnow().isoformat()
        }
        supabase.table("messages").insert(ai_msg_data).execute()

    return StreamingResponse(stream_response(), media_type="text/plain")

class BranchCreateRequest(BaseModel):
    name: str
    parent_message_id: str | None = None

@app.post("/chats/{chat_id}/branches")
def create_branch(
    chat_id: str,
    req: BranchCreateRequest,
    user_id: str = Depends(get_authenticated_user)
):
    """
    Creates a new branch within a given chat.
    Optionally references a 'parent_message_id' to show where it branched off.
    """
    # Validate chat ownership if you want
    chat_resp = supabase.table("chats").select("*").eq("id", chat_id).execute()
    if not chat_resp.data:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Insert the new branch
    data = {
        "chat_id": chat_id,
        "parent_message_id": req.parent_message_id,
        "name": req.name,
        "created_at": datetime.utcnow().isoformat(),
    }
    branch_resp = supabase.table("branches").insert(data).execute()
    if not branch_resp.data:
        raise HTTPException(status_code=500, detail="Failed to create branch")

    branch_id = branch_resp.data[0]["id"]
    return {"branch_id": branch_id, "name": req.name}


@app.get("/chats/{chat_id}/branches/{branch_id}/messages")
def get_branch_messages(
    chat_id: str,
    branch_id: str,
    user_id: str = Depends(get_authenticated_user),
):
    """
    Return all messages for a specific branch, sorted oldest to newest.
    """
    msg_resp = supabase.table("messages") \
                       .select("*") \
                       .eq("chat_id", chat_id) \
                       .eq("branch_id", branch_id) \
                       .order("created_at", desc=False) \
                       .execute()
    return msg_resp.data


from typing import Optional
class BranchFromRequest(BaseModel):
    name: Optional[str] = None
    tags: Optional[str] = None

@app.post("/chats/{chat_id}/branch-from/{message_id}")
def branch_chat(chat_id: str, message_id: str, req: BranchFromRequest, user_id: str = Depends(get_authenticated_user)):
    """
    Creates a branched chat by duplicating all messages from the original chat
    up to (and including) the specified parent message.
    The new chat is marked with 'branch_of' and 'branch_of_message_id' for traceability.
    """
    # 1. Retrieve the original chat
    chat_resp = supabase.table("chats").select("*").eq("id", chat_id).execute()
    if not chat_resp.data:
        raise HTTPException(status_code=404, detail="Original chat not found")
    original_chat = chat_resp.data[0]
    
    # 2. Retrieve the parent message
    parent_resp = supabase.table("messages").select("*").eq("id", message_id).execute()
    if not parent_resp.data:
        raise HTTPException(status_code=404, detail="Parent message not found")
    parent_message = parent_resp.data[0]
    parent_created_at = parent_message["created_at"]
    
    # 3. Retrieve all messages from original chat up to (and including) the parent message
    messages_resp = supabase.table("messages") \
                           .select("*") \
                           .eq("chat_id", chat_id) \
                           .lte("created_at", parent_created_at) \
                           .order("created_at", desc=False) \
                           .execute()
    messages_to_duplicate = messages_resp.data or []
    
    # 4. Generate new chat title using provided name if available
    new_title = req.name if req.name else f"Branch of {original_chat['title']}"
    
    # 5. Create a new chat as a branch
    new_chat_data = {
         "user_id": user_id,
         "title": new_title,
         "created_at": datetime.utcnow().isoformat(),
         "branch_of": chat_id,             # Ensure your chats table has this column
         "branch_of_message_id": message_id  # Ensure your chats table has this column
    }
    new_chat_resp = supabase.table("chats").insert(new_chat_data).execute()
    if not new_chat_resp.data:
         raise HTTPException(status_code=500, detail="Failed to create branched chat")
    new_chat_id = new_chat_resp.data[0]["id"]
    
    # 6. Duplicate each message into the new chat
    for msg in messages_to_duplicate:
         # Remove the existing 'id' so that a new one is generated upon insertion
         msg.pop("id", None)
         msg["chat_id"] = new_chat_id
         # Optionally, you might want to remove branch info if present in the original
         supabase.table("messages").insert(msg).execute()
    
    return {"new_chat_id": new_chat_id, "title": new_title}

class BranchMessageRequest(BaseModel):
    content: str

@app.post("/chats/{chat_id}/branches/{branch_id}/messages")
def create_branch_message(
    chat_id: str,
    branch_id: str,
    req: BranchMessageRequest,
    user_id: str = Depends(get_authenticated_user)
):
    """
    1) Insert a user message into this branch,
    2) Generate an AI reply (context from only this branch's messages),
    3) Insert the AI reply,
    4) Return both new messages.
    """
    # 1. Insert the user's message
    user_msg_data = {
        "chat_id": chat_id,
        "branch_id": branch_id,
        "user_id": user_id,
        "role": "user",
        "content": req.content,
        "created_at": datetime.utcnow().isoformat()
    }
    user_insert = supabase.table("messages").insert(user_msg_data).execute()
    if not user_insert.data:
        raise HTTPException(status_code=500, detail="Failed to insert user message")

    # 2. Build context from messages in *this branch* only
    context_resp = supabase.table("messages") \
                           .select("*") \
                           .eq("chat_id", chat_id) \
                           .eq("branch_id", branch_id) \
                           .order("created_at", desc=False) \
                           .limit(100) \
                           .execute()
    all_branch_msgs = context_resp.data or []
    context_prompt = ""
    for m in all_branch_msgs:
        context_prompt += f"{m['role']}: {m['content']}\n"


    # 3. Generate the AI reply
    client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))
    chat_obj = client.chats.create(model="gemini-2.0-flash")
    ai_response = chat_obj.send_message(context_prompt)
    ai_reply = ai_response.text.strip()

    # 4. Insert the AI's reply
    ai_msg_data = {
        "chat_id": chat_id,
        "branch_id": branch_id,
        "user_id": user_id,  # or a special value like "assistant"
        "role": "assistant",
        "content": ai_reply,
        "created_at": datetime.utcnow().isoformat()
    }
    ai_insert = supabase.table("messages").insert(ai_msg_data).execute()
    if not ai_insert.data:
        raise HTTPException(status_code=500, detail="Failed to insert AI message")

    return {
        "user_message": user_insert.data[0],
        "assistant_message": ai_insert.data[0]
    }

@app.get("/test")
def test(user_id: str = Depends(get_authenticated_user)):
    return {"user_id": user_id}

@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str, user_id: str = Depends(get_authenticated_user)):
    """
    Delete a chat and all its messages in the correct order to avoid foreign key errors.
    """
    # First verify the chat belongs to the user
    chat_resp = supabase.table("chats").select("*").eq("id", chat_id).eq("user_id", user_id).execute()
    if not chat_resp.data:
        raise HTTPException(status_code=404, detail="Chat not found or unauthorized")

    # Delete all branches of this chat first
    branch_chats = supabase.table("chats").select("id").eq("branch_of", chat_id).execute()
    if branch_chats.data:
        for branch in branch_chats.data:
            # Recursively delete each branch
            delete_chat(branch["id"], user_id)

    # Delete all branches referencing messages in this chat
    supabase.table("branches").delete().eq("chat_id", chat_id).execute()

    # Now delete all messages in the chat
    supabase.table("messages").delete().eq("chat_id", chat_id).execute()

    # Finally delete the chat itself
    supabase.table("chats").delete().eq("id", chat_id).execute()

    return {"status": "success"}

#uvicorn main:app --reload