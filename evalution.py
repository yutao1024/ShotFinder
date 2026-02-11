import os
import json
import argparse
import sys

def get_category(item):
    """
    Determine the category based on the 'Description' field of the input data.
    Logic is based on format_user_input in tools_lib.py.
    """
    desc = item.get('Description', '')
    
    if not desc:
        return "Shot"  # Default to Shot
    
    # Standardize processing
    desc_str = str(desc).strip()
    
    if "Temporal" in desc_str:
        return "Temporal"
    elif "Color" in desc_str:
        return "Color"
    elif "Style" in desc_str:
        return "Style"
    elif "Resolution" in desc_str:
        return "Resolution"
    elif "Audio" in desc_str:
        return "Audio"
    else:
        return "Shot"

def main():
    parser = argparse.ArgumentParser(description="Manual Evaluation for VLM Grounding Results")
    
    # 1. Define command-line arguments
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model being evaluated (e.g., gemini-1.5-pro)')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the input dataset JSON file (e.g., ./data_json/data.json)')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory containing the result JSON files (e.g., ./results/vlm_test)')
    parser.add_argument('--output_file', type=str, default=None, help='Path to save the evaluation report (e.g., ./results/eval_report.txt)')
    
    args = parser.parse_args()

    # 2. Validate paths
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file not found at {args.dataset_path}")
        sys.exit(1)
        
    if not os.path.exists(args.result_dir):
        print(f"Error: Result directory not found at {args.result_dir}")
        sys.exit(1)

    # 3. Load dataset
    try:
        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Loaded dataset: {len(dataset)} items.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # 4. Initialize statistics dictionary
    categories = ["Shot", "Temporal", "Color", "Style", "Resolution", "Audio"]
    stats = {cat: {'correct': 0, 'total': 0} for cat in categories}
    
    total_correct = 0
    total_count = 0
    missing_results = 0

    # 5. Iterate and calculate statistics
    print("Processing results...")
    
    for item in dataset:
        vid_id = item.get('id')
        category = get_category(item)
        
        # Path to the result file
        res_file_path = os.path.join(args.result_dir, f"{vid_id}.json")
        
        is_correct = False
        
        if os.path.exists(res_file_path):
            try:
                with open(res_file_path, 'r', encoding='utf-8') as f:
                    res_data = json.load(f)
                    # Key field: pass
                    if res_data.get('pass') is True:
                        is_correct = True
            except Exception:
                print(f"Warning: Failed to parse result for {vid_id}")
        else:
            # If result file is missing, treat as Fail and record it
            missing_results += 1

        # Update category statistics
        if category in stats:
            stats[category]['total'] += 1
            if is_correct:
                stats[category]['correct'] += 1
        
        # Update overall statistics (Avg)
        total_count += 1
        if is_correct:
            total_correct += 1

    # 6. Generate Report Content
    lines = []
    lines.append("="*85)
    lines.append(f" EVALUATION RESULTS | Model: {args.model_name}")
    lines.append("="*85)
    lines.append(f"{'Metric':<15} | {'Correct':<10} | {'Total':<10} | {'Accuracy (%)':<15}")
    lines.append("-" * 85)

    # Add metrics for each category
    for cat in categories:
        data = stats[cat]
        correct = data['correct']
        total = data['total']
        acc = (correct / total * 100) if total > 0 else 0.0
        
        acc_str = f"{acc:.2f}%" if total > 0 else "N/A"
        lines.append(f"{cat:<15} | {correct:<10} | {total:<10} | {acc_str:<15}")

    lines.append("-" * 85)
    
    # Add Average (Micro Average)
    avg_acc = (total_correct / total_count * 100) if total_count > 0 else 0.0
    lines.append(f"{'Avg.':<15} | {total_correct:<10} | {total_count:<10} | {avg_acc:.2f}%")
    lines.append("="*85)

    if missing_results > 0:
        lines.append(f"\n[Note] {missing_results} videos were counted as 'Fail' because their result files were missing.")

    # 7. Print to Console
    report_text = "\n".join(lines)
    print("\n" + report_text)

    # 8. Save to File (if specified)
    if args.output_file:
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(args.output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n[Success] Evaluation report saved to: {args.output_file}")
        except Exception as e:
            print(f"\n[Error] Failed to save report to file: {e}")

if __name__ == "__main__":
    main()


# python eval_manual.py \
#   --model_name "Gemini-1.5-Pro" \
#   --dataset_path "./data_json/video_dataset.json" \
#   --result_dir "./results/vlm_test" \
#   --output_file "./results/final_report.txt"