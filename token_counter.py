import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    """
    Counts the number of tokens in a given text using the tokenizer for the specified model.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def calculate_price(num_input_tokens, num_output_tokens, model="gpt-3.5-turbo"):
    """
    Calculates the estimated price in USD based on input/output token counts and the model used.
    """
    if model == "gpt-3.5-turbo":
        price_input_per_1k = 0.0005
        price_output_per_1k = 0.001
    elif model in ["gpt-4-turbo", "gpt-4o"]:
        price_input_per_1k = 0.01
        price_output_per_1k = 0.03
    else:
        raise ValueError(f"Unsupported model for pricing: {model}")


    cost = (num_input_tokens / 1000) * price_input_per_1k + (num_output_tokens / 1000) * price_output_per_1k
    return round(cost, 5)

def print_token_report(input_tokens, output_tokens, model="gpt-3.5-turbo"):
    """
    Prints a summary of token usage and estimated cost.
    """
    total_cost = calculate_price(input_tokens, output_tokens, model)
    print("\n📊 Usage details:")
    print(f"- Input tokens:  {input_tokens}")
    print(f"- Output tokens: {output_tokens}")
    print(f"- Total cost:    ${total_cost} USD (model: {model})")
