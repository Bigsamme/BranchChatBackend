from fastapi import FastAPI, Header, HTTPException, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import jwt
import json
from pydantic import BaseModel
from datetime import datetime
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from ai_providers import generate_title, generate_response, stream_response

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
    Skips signature verification for brevity; you should validate properly in production.
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

@app.post("/chats")
def create_empty_chat(user_id: str = Depends(get_authenticated_user)):
    """
    Creates a new chat with a placeholder title.
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
    Return all messages for a specific chat.
    """
    msg_resp = supabase.table("messages") \
                       .select("*") \
                       .eq("chat_id", chat_id) \
                       .order("created_at", desc=False) \
                       .execute()
    return msg_resp.data

@app.post("/chats/{chat_id}/messages")
def create_message(
    chat_id: str,
    req: MessageRequest,
    provider: str = "gemini",
    model: str = "gemini-2.0-flash",
    user_id: str = Depends(get_authenticated_user)
):
    # Check if the user has enough tokens
    user_resp = supabase.table("users").select("token_count").eq("user_id", user_id).single().execute()
    if user_resp.data["token_count"] <= 0:
        raise HTTPException(status_code=403, detail="You are out of credits. Please upgrade your plan to continue.")

    # 1. Insert the user's message
    user_msg_data = {
        "chat_id": chat_id,
        "user_id": user_id,
        "role": "user",
        "content": req.content,
        "created_at": datetime.utcnow().isoformat(),
        "model": model
    }
    user_insert = supabase.table("messages").insert(user_msg_data).execute()
    if not user_insert.data:
        raise HTTPException(status_code=500, detail="Failed to insert user message")
    
    # 2. Possibly rename the chat if it's still a placeholder and deduct tokens for title generation
    chat_resp = supabase.table("chats").select("*").eq("id", chat_id).execute()
    if not chat_resp.data:
        raise HTTPException(status_code=404, detail="Chat not found")
    current_title = chat_resp.data[0]["title"]
    if current_title == "Untitled Chat":
        new_title = generate_title(req.content, provider, model)
        supabase.table("chats").update({"title": new_title}).eq("id", chat_id).execute()
    
    # 3. Stream the AI reply, deduct tokens after streaming is complete, and return the stream
    def stream_with_deduction():
        full_reply = ""
        for chunk in stream_response(supabase, chat_id, user_id, provider, model):
            full_reply += chunk
            yield chunk

    return StreamingResponse(stream_with_deduction(), media_type="text/plain")

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
    chat_resp = supabase.table("chats").select("*").eq("id", chat_id).execute()
    if not chat_resp.data:
        raise HTTPException(status_code=404, detail="Chat not found")
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
    Return all messages for a specific branch.
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
    """
    chat_resp = supabase.table("chats").select("*").eq("id", chat_id).execute()
    if not chat_resp.data:
        raise HTTPException(status_code=404, detail="Original chat not found")
    original_chat = chat_resp.data[0]
    
    parent_resp = supabase.table("messages").select("*").eq("id", message_id).execute()
    if not parent_resp.data:
        raise HTTPException(status_code=404, detail="Parent message not found")
    parent_message = parent_resp.data[0]
    parent_created_at = parent_message["created_at"]
    
    # Only include messages up to the parent if it is not a user message
    query = supabase.table("messages") \
                    .select("*") \
                    .eq("chat_id", chat_id) \
                    .order("created_at", desc=False)

    if parent_message["role"] == "user":
        query = query.lt("created_at", parent_created_at)
    else:
        query = query.lte("created_at", parent_created_at)

    messages_resp = query.execute()
    messages_to_duplicate = messages_resp.data or []
    
    new_title = req.name if req.name else f"Branch of {original_chat['title']}"
    
    new_chat_data = {
         "user_id": user_id,
         "title": new_title,
         "created_at": datetime.utcnow().isoformat(),
         "branch_of": chat_id,
         "branch_of_message_id": message_id
    }
    new_chat_resp = supabase.table("chats").insert(new_chat_data).execute()
    if not new_chat_resp.data:
         raise HTTPException(status_code=500, detail="Failed to create branched chat")
    new_chat_id = new_chat_resp.data[0]["id"]
    
    for msg in messages_to_duplicate:
         msg.pop("id", None)
         msg["chat_id"] = new_chat_id
         supabase.table("messages").insert(msg).execute()
    
    return {"new_chat_id": new_chat_id, "title": new_title}

class BranchMessageRequest(BaseModel):
    content: str

@app.post("/chats/{chat_id}/branches/{branch_id}/messages")
def create_branch_message(
    chat_id: str,
    branch_id: str,
    req: BranchMessageRequest,
    provider: str = "gemini",
    model: str = "gemini-2.0-flash",
    user_id: str = Depends(get_authenticated_user)
):
    """
    1) Insert a user message into this branch.
    2) Generate an AI reply using context from this branch and the selected provider/model.
    3) Insert the AI reply and return both messages.
    """
    user_msg_data = {
        "chat_id": chat_id,
        "branch_id": branch_id,
        "user_id": user_id,
        "role": "user",
        "content": req.content,
        "created_at": datetime.utcnow().isoformat(),
        "model": model
    }
    user_insert = supabase.table("messages").insert(user_msg_data).execute()
    if not user_insert.data:
        raise HTTPException(status_code=500, detail="Failed to insert user message")

    context_resp = supabase.table("messages") \
                           .select("*") \
                           .eq("chat_id", chat_id) \
                           .eq("branch_id", branch_id) \
                           .order("created_at", desc=False) \
                           .limit(10) \
                           .execute()
    all_branch_msgs = context_resp.data or []
    context_prompt = ""
    for m in all_branch_msgs:
        context_prompt += f"{m['role']}: {m['content']}\n"

    response = generate_response(context_prompt, provider, model)
    ai_reply = response["content"]
    
    ai_msg_data = {
        "chat_id": chat_id,
        "branch_id": branch_id,
        "user_id": user_id,
        "role": "assistant",
        "content": ai_reply,
        "created_at": datetime.utcnow().isoformat(),
        "model": model
    }
    ai_insert = supabase.table("messages").insert(ai_msg_data).execute()
    if not ai_insert.data:
        raise HTTPException(status_code=500, detail="Failed to insert AI message")
    
    # Estimate and deduct token cost

    return {
        "user_message": user_insert.data[0],
        "assistant_message": ai_insert.data[0]
    }

@app.get("/test")
def test(user_id: str = Depends(get_authenticated_user)):
    return {"user_id": user_id}

@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str, user_id: str = Depends(get_authenticated_user)):
    chat_resp = supabase.table("chats").select("*").eq("id", chat_id).eq("user_id", user_id).execute()
    if not chat_resp.data:
        raise HTTPException(status_code=404, detail="Chat not found or unauthorized")
    branch_chats = supabase.table("chats").select("id").eq("branch_of", chat_id).execute()
    if branch_chats.data:
        for branch in branch_chats.data:
            delete_chat(branch["id"], user_id)
    supabase.table("branches").delete().eq("chat_id", chat_id).execute()
    supabase.table("messages").delete().eq("chat_id", chat_id).execute()
    supabase.table("chats").delete().eq("id", chat_id).execute()
    return {"status": "success"}


@app.post("/api/user")
async def user(request: Request):
    """Handles Clerk webhook when a user is created. Saves the user_id and sets token_count to 10000 in Supabase."""
    payload = await request.json()
    user_data = payload.get("data")
    user_id = user_data.get("id") if user_data else None
    if not user_id:
        raise HTTPException(status_code=400, detail="User id not found in payload")
    
    # Insert a new record into the 'users' table with a token_count of 10000
    data = {
        "user_id": user_id,
        "token_count": 10000,
        "created_at": datetime.utcnow().isoformat()
    }
    response = supabase.table("users").insert(data).execute()
    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to create user record")
    
    return {"status": "success", "user_id": user_id}

import stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")  # add to your .env

@app.post("/create-checkout-session")
async def create_checkout_session(
    user_id: str = Depends(get_authenticated_user),
    body: dict = Body(...)
):
    try:
        plan = body.get("plan")
        price_lookup = {
            "starter": "price_1R9ZFeE8gTxt6YIVJCZMAPUK",
            "pro": "price_1R9ZFzE8gTxt6YIVxHBZFg1Y",
            "power": "price_1R9ZGCE8gTxt6YIV79ruFhpn"
        }
        price_id = price_lookup.get(plan)
        if not price_id:
            raise HTTPException(status_code=400, detail="Invalid plan")

        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url="http://localhost:3000/success",
            cancel_url="http://localhost:3000/cancel",
            client_reference_id=user_id,
            subscription_data={"metadata": {"user_id": user_id, "plan": plan}}
        )

        return {"url": checkout_session.url}
    except Exception as e:
        print("Stripe error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-portal-session")
async def create_portal_session(user_id: str = Depends(get_authenticated_user)):
    try:
        # Get the user's stored subscription ID
        user = supabase.table("users").select("subscription_id").eq("user_id", user_id).single().execute()
        subscription_id = user.data.get("subscription_id")

        if not subscription_id:
            raise HTTPException(status_code=404, detail="Subscription ID not found")

        # Retrieve the customer ID from the subscription
        subscription = stripe.Subscription.retrieve(subscription_id)
        customer_id = subscription.get("customer")

        # Create a portal session
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url="http://localhost:3000/dashboard"
        )
        return {"url": session.url}

    except Exception as e:
        print("Stripe portal error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle the checkout.session.completed event
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session.get("client_reference_id")
        subscription_id = session.get("subscription")
        subscription = stripe.Subscription.retrieve(session.get("subscription"))
        metadata = subscription.get("metadata", {})
        plan = metadata.get("plan")
        print(plan)

        token_grants = {
            "starter": 100000,
            "pro": 500000,
            "power": 5000000
        }
        token_count = token_grants.get(plan)


        if user_id and subscription_id:
            supabase.table("users").update({
                "token_count": token_count or 10000,
                "subscription_id": subscription_id
            }).eq("user_id", user_id).execute()
            supabase.table("users").update({
                "plan": plan
            }).eq("user_id", user_id).execute()
    
    elif event["type"] == "invoice.payment_succeeded":
        invoice = event["data"]["object"]
        subscription_id = invoice.get("subscription")

        if not subscription_id:
            print("No subscription ID in invoice.payment_succeeded payload. Skipping.")
            return {"status": "skipped"}

        subscription = stripe.Subscription.retrieve(subscription_id)
        metadata = subscription.get("metadata", {})
        user_id = metadata.get("user_id")
        plan = metadata.get("plan")

        token_grants = {
            "starter": 100000,
            "pro": 500000,
            "power": 5000000
        }
        token_count = token_grants.get(plan)

        

        if user_id and token_count:
            supabase.table("users").update({
                "token_count": token_count
            }).eq("user_id", user_id).execute()


    return {"status": "success"}



@app.get("/user/token_count")
def get_token_count(user_id: str = Depends(get_authenticated_user)):
    response = supabase.table("users").select("token_count, plan").eq("user_id", user_id).single().execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="User not found")
    print(response.data["plan"], "hello")
    print(response.data["token_count"], "hello")

    return {
        "token_count": response.data["token_count"],
        "plan": response.data["plan"]
    }


# uvicorn main:app --reload
