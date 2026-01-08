import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results(csv_path, output_image="lstm_prediction_plot.png", target_points=5000):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Identify columns
    # We look for columns starting with 'actual_' and 'pred_'
    actual_cols = [c for c in df.columns if c.startswith('actual_')]
    pred_cols = [c for c in df.columns if c.startswith('pred_')]
    
    if not actual_cols or not pred_cols:
        print("Error: Could not find actual/pred columns in CSV.")
        print(f"Columns found: {df.columns.tolist()}")
        return

    # Convert timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_col = df['timestamp']
    else:
        time_col = df.index

    # Sampling logic
    total_points = len(df)
    stride = 1
    if total_points > target_points:
        stride = total_points // target_points
        print(f"Data has {total_points} points. Downsampling by factor of {stride} (plotting ~{target_points} points).")
    
    df_sampled = df.iloc[::stride]
    time_sampled = time_col.iloc[::stride]

    # Plotting
    num_features = len(actual_cols)
    fig, axes = plt.subplots(num_features, 1, figsize=(15, 5 * num_features), sharex=True)
    
    if num_features == 1:
        axes = [axes]

    for i, (act_col, pred_col) in enumerate(zip(actual_cols, pred_cols)):
        ax = axes[i]
        
        # Plot Actual
        ax.plot(time_sampled, df_sampled[act_col], label='Actual', color='blue', alpha=0.6, linewidth=1)
        
        # Plot Predicted
        ax.plot(time_sampled, df_sampled[pred_col], label='Predicted', color='orange', alpha=0.7, linewidth=1, linestyle='--')
        
        # Highlight Anomalies if label exists
        if 'label' in df.columns:
            anomalies = df_sampled[df_sampled['label'] == 1]
            if not anomalies.empty:
                # Plot anomalies as red dots on the actual line
                ax.scatter(anomalies['timestamp'] if 'timestamp' in df.columns else anomalies.index, 
                           anomalies[act_col], color='red', label='Anomaly Label', zorder=5, s=10)

        ax.set_title(f"Actual vs Predicted: {act_col.replace('actual_', '')}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Value")

    plt.xlabel("Time")
    plt.tight_layout()
    
    print(f"Saving plot to {output_image}...")
    plt.savefig(output_image)
    print("Done.")

if __name__ == "__main__":
    plot_results("lstm_ad_results.csv")
