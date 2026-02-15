"""
Convert UIGEN ChatML format to simple prompt→code format for base model training.

Converts from:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

To:
    {"text": "# Task: Generate HTML/CSS code\n# Requirements: <user_content>\n\n<assistant_code>"}

Usage:
    python finetuning/convert_uigen_to_base_format.py
"""

import argparse
import json
import os


def convert_chatml_to_base_format(item: dict) -> dict:
    """Convert ChatML conversational format OR raw format to simple prompt→code format.
    
    Args:
        item: Dict with either:
              - 'messages' key containing list of message dicts (ChatML format)
              - 'question' and 'answer' keys (raw format for test data)
        
    Returns:
        Dict with 'text' key containing formatted prompt and code
    """
    user_content = ""
    assistant_content = ""
    
    # Check format: ChatML (messages) or raw (question/answer)
    if 'messages' in item:
        # ChatML format
        messages = item['messages']
        
        # Extract user and assistant messages
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'user':
                user_content = content
            elif role == 'assistant':
                assistant_content = content
    
    elif 'question' in item and 'answer' in item:
        # Raw format (test data)
        user_content = item['question']
        assistant_content = item['answer']
    
    else:
        # Fallback: try to extract any text
        user_content = item.get('text', str(item))
        assistant_content = ""
    
    # Format as simple prompt→code
    # Using a minimal prefix that helps the model understand the task
    formatted_text = f"# Task: Generate HTML/CSS code using Tailwind CSS\n# Requirements: {user_content}\n\n{assistant_content}"
    
    return {"text": formatted_text}


def convert_file(input_path: str, output_path: str) -> int:
    """Convert a JSONL file from ChatML to base format.
    
    Args:
        input_path: Path to input JSONL file in ChatML format
        output_path: Path to output JSONL file in base format
        
    Returns:
        Number of samples converted
    """
    converted_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            item = json.loads(line.strip())
            converted_item = convert_chatml_to_base_format(item)
            f_out.write(json.dumps(converted_item, ensure_ascii=False) + '\n')
            converted_count += 1
    
    return converted_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert UIGEN ChatML format to base model format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data",
        help="Input directory containing ChatML JSONL files (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/base_format",
        help="Output directory for base format JSONL files (default: data/base_format)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["uigen_train.jsonl", "uigen_val.jsonl", "uigen_test.jsonl"],
        help="Files to convert (default: uigen_train.jsonl uigen_val.jsonl uigen_test.jsonl)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Converting UIGEN ChatML format to Base Model format")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    total_converted = 0
    
    for filename in args.files:
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)
        
        if not os.path.exists(input_path):
            print(f"⚠️  Skipping {filename} (not found)")
            continue
        
        print(f"Converting {filename}...", end=" ")
        count = convert_file(input_path, output_path)
        total_converted += count
        print(f"✓ {count} samples")
    
    print()
    print("=" * 80)
    print(f"Conversion complete! Total samples converted: {total_converted}")
    print("=" * 80)
    print()
    print("Example of converted format:")
    print("-" * 80)
    
    # Show example from first output file
    first_output = os.path.join(args.output_dir, args.files[0])
    if os.path.exists(first_output):
        with open(first_output, 'r', encoding='utf-8') as f:
            first_item = json.loads(f.readline())
            example_text = first_item['text']
            # Show first 500 chars
            print(example_text[:500] + ("..." if len(example_text) > 500 else ""))
    
    print("-" * 80)
    print()
    print("You can now use these files with the base model training script:")
    print(f"  modal run finetuning/unsloth/modal_qwen_code_generation.py \\")
    print(f"    --dataset-path {args.output_dir}/uigen_train.jsonl \\")
    print(f"    --num-epochs 3")
    print()


if __name__ == "__main__":
    main()
