"""Visualization script for federated learning results."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os

# Set style for better visualizations
plt.style.use('default')  # Use default style
sns.set_theme()  # Apply seaborn defaults

def parse_results_file(file_path):
    """Parse the results file and extract metrics."""
    summaries = []
    current_summary = None
    in_evaluate_section = False
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Start of new summary section
        if '[SUMMARY' in line:
            if current_summary:
                summaries.append(current_summary)
            current_summary = {
                'name': line.split('[SUMMARY - ')[-1].strip(']') if ' - ' in line else 'Unknown',
                'distributed_loss': [],
                'train_loss': [],
                'accuracy': []
            }
            in_evaluate_section = False
        
        # Track evaluate section
        elif "History (metrics, distributed, evaluate):" in line:
            in_evaluate_section = True
            
            # Look ahead for accuracy data
            acc_data = []
            j = i + 1
            while j < len(lines) and "History" not in lines[j]:
                if "'accuracy': [" in lines[j]:
                    # Start collecting accuracy data
                    acc_str = ""
                    while j < len(lines) and "]}" not in lines[j]:
                        if "'accuracy': [" in lines[j]:
                            acc_str += lines[j].split("'accuracy': [")[1].strip()
                        else:
                            acc_str += lines[j].strip()
                        j += 1
                    if j < len(lines):
                        acc_str += lines[j].split("]}")[0]
                    
                    try:
                        # Clean up the string and parse manually
                        acc_str = acc_str.replace("\n", "").replace(" ", "")
                        # Remove INFO: markers and other non-data text
                        acc_str = acc_str.replace("INFO:", "")
                        
                        # Parse the tuples manually
                        pairs = []
                        for pair in acc_str.split("),("):
                            pair = pair.strip("()").split(",")
                            round_num = int(pair[0])
                            acc_val = float(pair[1])
                            pairs.append((round_num, acc_val))
                        
                        if current_summary is not None:
                            current_summary['accuracy'] = pairs
                            print(f"\nSuccessfully parsed {len(pairs)} accuracy values:")
                            print(f"First: {pairs[0]}")
                            print(f"Last: {pairs[-1]}")
                    except Exception as e:
                        print(f"\nDebug - Accuracy parsing:")
                        print(f"Raw string: {acc_str}")
                        print(f"Error: {str(e)}")
                j += 1
            i = j - 1  # Adjust main counter to skip processed lines
            
        # Parse distributed loss
        elif 'round' in line and ':' in line and not 'History' in line:
            try:
                parts = line.split("round")[-1].split(":")
                round_num = int(parts[0].strip())
                loss_val = float(parts[1].strip())
                if current_summary is not None:
                    current_summary['distributed_loss'].append((round_num, loss_val))
            except:
                pass
        
        i += 1
    
    # Add the last summary
    if current_summary:
        summaries.append(current_summary)
    
    # Validate parsed data
    for summary in summaries:
        print(f"\nValidating data for: {summary['name']}")
        print(f"- Found {len(summary['distributed_loss'])} distributed loss entries")
        print(f"- Found {len(summary['accuracy'])} accuracy entries")
        if summary['accuracy']:
            print(f"- Accuracy range: {min(v for _, v in summary['accuracy']):.4f} to {max(v for _, v in summary['accuracy']):.4f}")
    
    return summaries

def plot_metrics(summary):
    """Create a comprehensive visualization of all metrics."""
    # Validate data
    if not summary['distributed_loss'] and not summary['accuracy']:
        print(f"Warning: No data to plot for {summary['name']}")
        return None
        
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Loss Curves
    if summary['distributed_loss']:
        rounds, losses = zip(*sorted(summary['distributed_loss']))
        ax1.plot(rounds, losses, 'o-', linewidth=2, label='Distributed Loss', color='#FF6B6B')
        
        # Add value annotations to loss curves
        for i, loss in enumerate(losses):
            ax1.annotate(f'{loss:.3f}', (rounds[i], loss), textcoords="offset points", 
                        xytext=(0,10), ha='center')
                        
        # Set y-axis limits with padding
        loss_min, loss_max = min(losses), max(losses)
        padding = (loss_max - loss_min) * 0.1
        ax1.set_ylim([loss_min - padding, loss_max + padding])
    
    if summary['train_loss']:
        train_rounds, train_losses = zip(*sorted(summary['train_loss']))
        if any(loss != 0 for loss in train_losses):  # Only plot if we have non-zero values
            ax1.plot(train_rounds, train_losses, 's-', linewidth=2, label='Training Loss', color='#4ECDC4')
            for i, loss in enumerate(train_losses):
                ax1.annotate(f'{loss:.3f}', (train_rounds[i], loss), textcoords="offset points", 
                            xytext=(0,-15), ha='center')
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Progression Over Rounds')
    ax1.grid(True, alpha=0.3)
    if ax1.get_legend_handles_labels()[0]:  # Only add legend if we have data
        ax1.legend()
    
    # Plot 2: Accuracy Curve
    if summary['accuracy']:
        acc_rounds, accuracy = zip(*sorted(summary['accuracy']))
        if any(acc != 0 for acc in accuracy):  # Only plot if we have non-zero values
            ax2.plot(acc_rounds, accuracy, 'D-', linewidth=2, color='#45B7D1', label='Accuracy')
            ax2.fill_between(acc_rounds, 
                           [a-0.02 for a in accuracy],  # Reduced confidence interval
                           [a+0.02 for a in accuracy], 
                           alpha=0.2, color='#45B7D1')
            
            # Add value annotations to accuracy curve
            for i, acc in enumerate(accuracy):
                ax2.annotate(f'{acc:.1%}', (acc_rounds[i], acc), textcoords="offset points", 
                            xytext=(0,10), ha='center')
            
            # Set y-axis limits with padding
            acc_min, acc_max = min(accuracy), max(accuracy)
            padding = (acc_max - acc_min) * 0.1
            ax2.set_ylim([max(0, acc_min - padding), min(1, acc_max + padding)])
    
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Accuracy Progression')
    ax2.grid(True, alpha=0.3)
    if ax2.get_legend_handles_labels()[0]:  # Only add legend if we have data
        ax2.legend()
    
    plt.suptitle(f"Federated Learning Metrics - {summary['name']}", y=1.05)
    plt.tight_layout()
    
    # Create filename from summary name (sanitized)
    safe_name = "".join(c for c in summary['name'] if c.isalnum() or c in (' ', '-', '_'))[:50]
    filename = f"federated_learning_metrics_{safe_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def plot_performance_summary(summary):
    """Create a summary visualization showing final performance."""
    # Validate data
    if not summary['distributed_loss'] or not summary['accuracy']:
        print(f"Warning: Insufficient data for performance summary of {summary['name']}")
        return None
        
    try:
        # Calculate improvements
        initial_loss = summary['distributed_loss'][0][1]
        final_loss = summary['distributed_loss'][-1][1]
        loss_improvement = (initial_loss - final_loss) / initial_loss * 100
        
        initial_acc = summary['accuracy'][0][1]
        final_acc = summary['accuracy'][-1][1]
        acc_improvement = (final_acc - initial_acc) / initial_acc * 100 if initial_acc > 0 else 0
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar plot
        metrics = ['Final Accuracy', 'Loss Reduction', 'Accuracy Improvement']
        values = [final_acc * 100, loss_improvement, acc_improvement]
        colors = ['#45B7D1', '#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(metrics, values, color=colors)
        ax.set_title(f'Performance Summary - {summary["name"]}')
        ax.set_ylabel('Percentage (%)')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Create filename from summary name (sanitized)
        safe_name = "".join(c for c in summary['name'] if c.isalnum() or c in (' ', '-', '_'))[:50]
        filename = f"performance_summary_{safe_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
        
    except Exception as e:
        print(f"Warning: Could not generate performance summary for {summary['name']}: {str(e)}")
        return None

def main():
    """Generate all visualizations."""
    # Get the path to results.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(os.path.dirname(script_dir), 'results.txt')
    
    if not os.path.exists(results_file):
        print(f"Error: Could not find results file at {results_file}")
        return
    
    print("Parsing results file...")
    summaries = parse_results_file(results_file)
    
    if not summaries:
        print("No data found in results file.")
        return
    
    print(f"\nGenerating visualizations for {len(summaries)} experiments...")
    for summary in summaries:
        print(f"\nProcessing experiment: {summary['name']}")
        
        # Data validation
        print("Data points found:")
        print(f"- Distributed Loss: {len(summary['distributed_loss'])} rounds")
        print(f"- Training Loss: {len(summary['train_loss'])} rounds")
        print(f"- Accuracy: {len(summary['accuracy'])} rounds")
        
        metrics_file = plot_metrics(summary)
        summary_file = plot_performance_summary(summary)
        
        if metrics_file or summary_file:
            print("\nFiles saved:")
            if metrics_file:
                print(f"1. {metrics_file}")
            if summary_file:
                print(f"2. {summary_file}")
        else:
            print("\nNo visualizations generated due to insufficient data")

if __name__ == "__main__":
    main() 