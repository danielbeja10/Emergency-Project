import os
from summarizer import summarize_medical_file, save_summary_to_file, load_medical_file
from config import OPENAI_API_KEY
from token_counter import count_tokens, print_token_report
import ModelSelector

# Define global variable for the chosen GPT model
SELECTED_MODEL = "gpt-4-turbo"



def main():
    global SELECTED_MODEL

    # Get and validate GPT model from user
    SELECTED_MODEL = ModelSelector.get_valid_model()

    # Ask the user if they want to run in simulation mode
    simulation_input = input("Run in simulation mode? (y/N): ").strip().lower()
    SIMULATION_MODE = simulation_input == "y"

    # Prompt for patient ID
    patient_id = input("Enter patient ID number (Teudat Zehut): ").strip()
    input_file = f"examples/{patient_id}.txt"
    output_file = f"output/summary_{patient_id}.txt"

    # Check if the patient file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' not found.")
        return

    # Load medical file and count input tokens (unless in simulation)
    print("⏳ Reading file and generating summary...")
    original_text = load_medical_file(input_file)
    input_tokens = count_tokens(original_text, SELECTED_MODEL) if not SIMULATION_MODE else 0

    # Generate summary using the specified model or mock
    summary = summarize_medical_file(input_file, SELECTED_MODEL, simulation=SIMULATION_MODE)
    output_tokens = count_tokens(summary, SELECTED_MODEL) if not SIMULATION_MODE else 0

    # Save the summary to an output file
    save_summary_to_file(summary, output_file)
    print(f"✅ Summary saved to '{output_file}'")

    # Print a token usage and cost report, or simulation notice
    if SIMULATION_MODE:
        print("⚠️  Running in simulation mode. No tokens were used and no API calls were made.")
    else:
        print_token_report(input_tokens, output_tokens, SELECTED_MODEL)

if __name__ == "__main__":
    main()
