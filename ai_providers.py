import os
from google import genai
from openai import OpenAI
import anthropic
from datetime import datetime
from fastapi import HTTPException
from token_utils import deduct_tokens



def generate_title(text: str, provider: str, model: str) -> str:
    prompt = f"Can you make this into a short title for a chat: {text}\nReturn only the title text and nothing else."
    return generate_response(prompt, provider, model)

def generate_response(prompt: str, provider: str, model: str) -> str:
    if provider.lower() == "gemini":
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        chat = client.chats.create(model=model)
        full_prompt = prompt
        response = chat.send_message_stream(full_prompt)
        return ''.join(chunk.text for chunk in response).strip()

    elif provider.lower() == "openai":
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        return ''.join(
            chunk.choices[0].delta.content or "" for chunk in response
        ).strip()

    elif provider.lower() == "claude":
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        raw_response = []
        with client.messages.stream(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        ) as stream:
            for text in stream.text_stream:
                raw_response.append(text)
                return ''.join(raw_response).strip()

    else:
        raise ValueError("Unsupported provider")
    

def stream_response(supabase, chat_id: str, user_id: str, provider: str, model: str):
    """Streams the AI response based on the context from the last 100 messages and stores the complete reply."""
    full_text = ""
    # Rebuild context from the last 100 messages
    msg_resp = supabase.table("messages") \
                       .select("*") \
                       .eq("chat_id", chat_id) \
                       .order("created_at", desc=True) \
                       .limit(100) \
                       .execute()
    messages_list = list(reversed(msg_resp.data or []))
    context_prompt = ""
    for message in messages_list:
        context_prompt += f"{message['role']}: {message['content']}\n"
    
    context_prompt = context_prompt

    if provider.lower() == "gemini":
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        chat_obj = client.chats.create(model=model)
        raw_response = []
        for chunk in chat_obj.send_message_stream(context_prompt):
            raw_response.append(chunk)
            full_text += chunk.text
            yield chunk.text
    elif provider.lower() == "openai":
        client = OpenAI()
        raw_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": context_prompt}
            ],
            stream=True,
            stream_options={"include_usage": True}
        )
        for chunk in raw_response:
            # Check for final chunk containing usage stats
            if chunk.choices == [] and getattr(chunk, "usage", None):
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
                print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
                raw_response = {"input_tokens": input_tokens, "output_tokens": output_tokens}
                break
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_text += content
                yield content
        
        
    elif provider.lower() == "claude":
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        raw_response = []
        stream = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": context_prompt}],
            max_tokens=64000,
            stream=True,
        )
        for event in stream:
            raw_response.append(event)
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                delta_text = event.delta.text
                full_text += delta_text
                yield delta_text


        start_event = raw_response[0]
        delta_event = raw_response[-2]

        # Input tokens from the start event
        input_tokens = getattr(getattr(start_event.message, "usage", None), "input_tokens", 0)

        # Use only the final output_tokens value from the delta event
        output_tokens = delta_event.usage.output_tokens

        raw_response = {"input_tokens":input_tokens, "output_tokens":output_tokens}
    else:
        raise HTTPException(status_code=400, detail="Unsupported provider")
    
    # Store the complete AI reply after streaming is done
    ai_msg_data = {
        "chat_id": chat_id,
        "user_id": user_id,
        "role": "assistant",
        "content": full_text,
        "created_at": datetime.utcnow().isoformat(),
        "model": model
    }
    supabase.table("messages").insert(ai_msg_data).execute()
    deduct_tokens(user_id, provider, model, raw_response, supabase)