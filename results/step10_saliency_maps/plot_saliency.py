import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

csv_files = glob.glob("saliency_*.csv")
print(f"Found {len(csv_files)} CSV files. Plotting in English...")

for f in csv_files:
    df = pd.read_csv(f)
    if len(df) == 0:
        continue
        
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Bar chart for transaction amount
    color = 'tab:blue'
    ax1.set_xlabel('Transaction Index')
    ax1.set_ylabel('Transaction Amount (Normalized)', color=color)
    ax1.bar(df['tx_global_idx'], df['z_amount'], color=color, alpha=0.6, label='ETH Amount')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Line chart for AI Attention Score
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Attention Score', color=color)  
    ax2.plot(df['tx_global_idx'], df['attention_score'], color=color, linewidth=2.5, label='Attention')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Grid
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    account_name = f.replace("saliency_", "").replace(".csv", "")
    plt.title(f'Saliency Map: AI Attention vs. Transaction Amount\nAccount: {account_name}')
    fig.tight_layout()
    
    png_name = f.replace(".csv", ".png")
    plt.savefig(png_name, dpi=300)
    plt.close()

print("Done! All plots regenerated in English.")
