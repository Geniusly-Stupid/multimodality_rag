
import json
import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics(log_path="training_log.json", output_path="training_metrics.png"):
    """
    Reads training logs from a JSON file and plots the training metrics.
    """
    try:
        with open(log_path, 'r') as f:
            log_history = json.load(f)
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        print("Please run the training script first to generate the log file.")
        return

    # Separate logs for training and evaluation
    train_logs = [log for log in log_history if 'loss' in log and 'epoch' in log]
    eval_logs = [log for log in log_history if 'eval_loss' in log and 'epoch' in log]

    if not train_logs and not eval_logs:
        print("No training or evaluation logs found in the log file.")
        return

    # Create a DataFrame for easier plotting
    df_train = pd.DataFrame(train_logs)
    df_eval = pd.DataFrame(eval_logs)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Training Metrics Over Epochs')

    # Plotting training and evaluation loss
    if not df_train.empty:
        ax1.plot(df_train['epoch'], df_train['loss'], label='Training Loss')
    if not df_eval.empty:
        ax1.plot(df_eval['epoch'], df_eval['eval_loss'], label='Evaluation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Evaluation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plotting evaluation MSE
    if not df_eval.empty:
        ax2.plot(df_eval['epoch'], df_eval['eval_mse'], label='Evaluation MSE', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_title('Evaluation MSE')
    ax2.legend()
    ax2.grid(True)

    # Save the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_metrics()
