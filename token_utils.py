from fastapi import HTTPException
import json




def calculate_token_cost(provider: str, model: str, response: str) -> int:
    """
    Calculates the token cost based on the relative pricing from each provider.
    """
    # OpenAI pricing per 1M tokens (in USD)
    openai_pricing = {
        "gpt-4o": {"input": 2.5, "output": 10},
        "gpt-4o-mini": {"input": 0.150, "output": 0.6},
    }
    
    # Claude pricing per 1M tokens (in USD)
    claude_pricing = {
        "claude-3-7-sonnet": {"input": 3, "output": 15},
        "claude-3-5-haiku": {"input": 0.8, "output": 4},
    }
    
    # Gemini pricing per 1M tokens (in USD)
    gemini_pricing = {
        "gemini-2.0-flash": {"input": 0.1, "output": 0.4},
        "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.3},
        "gemini-1.5-flash": {"input": 0.15, "output": 0.6},
        "gemini-1.5-flash-8b": {"input": 0.075, "output": 0.3},
        "gemini-1.5-pro": {"input": 1.25, "output": 7.5},
    }
    
    # Select the appropriate pricing model
    if provider.lower() == "openai":
        model_pricing = openai_pricing.get(model, {"input": 1.0, "output": 1.0})
    elif provider.lower() == "claude":
        model_pricing = claude_pricing.get(model, {"input": 1.0, "output": 1.0})
    elif provider.lower() == "gemini":
        model_pricing = gemini_pricing.get(model, {"input": 1.0, "output": 1.0})
    else:
        # Default pricing if provider not recognized
        model_pricing = {"input": 1.0, "output": 1.0}
    
    try:
        # Extract token counts from response
        if provider.lower() == "openai":
            input_tokens = response["input_tokens"]
            output_tokens = response["output_tokens"]
        elif provider.lower() == "claude":
            input_tokens = response["input_tokens"]
            output_tokens = response["output_tokens"]
        elif provider.lower() == "gemini":
            response_data = response
            if isinstance(response_data, list):
                usage_metadata = response_data[-1].usage_metadata
                input_tokens = usage_metadata.prompt_token_count if usage_metadata else 0
                output_tokens = usage_metadata.candidates_token_count if usage_metadata else 0
                print("Gemini token debug:")
                print("Prompt tokens:", input_tokens)
                print("Candidate tokens:", output_tokens)
            else:
                input_tokens = 0
                output_tokens = 0
        else:
            print("Warning: Unsupported provider. Cannot extract token usage.")
            return -1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return -1
    
    # Calculate the actual dollar cost
    input_cost = (model_pricing["input"] / 1000000) * input_tokens
    output_cost = (model_pricing["output"] / 1000000) * output_tokens
    total_cost = input_cost + output_cost
    
    
    # Scale to a more practical number for token allocation
    # Multiply by 1,000 to convert to millicents (thousandths of a cent)
    # This will give numbers in the range of hundreds or thousands for typical API calls
    scaled_tokens = int(total_cost * 1000000)
    print(scaled_tokens)
    
    return scaled_tokens

def deduct_tokens(user_id: str, provider: str, model: str, response: str, supabase):
    """Deducts tokens from a user's account based on provider and model."""
    tokens_to_deduct = calculate_token_cost(provider, model, response)

    # Handle potential error from calculate_token_cost
    if tokens_to_deduct == -1:
        print("Error: Could not calculate token cost.  Token deduction skipped.")
        return #Do not proceed with token deduction if token cost calculation failed.

    user_resp = supabase.table("users").select("token_count").eq("user_id", user_id).execute()
    if not user_resp.data:
        raise HTTPException(status_code=404, detail="User not found")
    current_tokens = user_resp.data[0]["token_count"]
    new_token_count = max(current_tokens - tokens_to_deduct, 0)
    supabase.table("users").update({"token_count": new_token_count}).eq("user_id", user_id).execute()