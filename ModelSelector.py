# List of supported models with display options
SUPPORTED_MODELS = ["gpt-4-turbo", "gpt-3.5-turbo", "gpt-4", "gpt-4o"]

def get_valid_model():
    """
    Prompt user to select a GPT model by entering 1, 2, 3, or 4.
    Returns the selected model as a string.
    """
    print("Select GPT model:")
    for idx, model in enumerate(SUPPORTED_MODELS, start=1):
        print(f"  {idx}. {model}")

    while True:
        selection = input("Enter the number of the desired model (default: 1): ").strip()
        if not selection:
            return SUPPORTED_MODELS[0]
        if selection.isdigit():
            index = int(selection) - 1
            if 0 <= index < len(SUPPORTED_MODELS):
                return SUPPORTED_MODELS[index]
        print("❌ Invalid input. Please enter a valid number.")
