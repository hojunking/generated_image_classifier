

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np

df = pd.read_csv('imagen_results.csv')

tn_samples = df[(df['label'] == 0) & (df['pred'] == 0)].sample(n=min(8, len(df[(df['label'] == 0) & (df['pred'] == 0)])), random_state=4)
fp_samples = df[(df['label'] == 0) & (df['pred'] == 1)].sample(n=min(8, len(df[(df['label'] == 0) & (df['pred'] == 1)])), random_state=4)


def visualize_and_save(samples, title_prefix, filename, rows=2, cols=4, fig_size=(20, 10)):
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    axes = np.array(axes).reshape(-1)  # Ensure axes is flat for consistent indexing
    title_set = False  # Flag to track if the title has been set
    
    for i, (idx, row) in enumerate(samples.iterrows()):
        img = mpimg.imread(row['path'])  # Replace with actual image loading
        axes[i].imshow(img)
        
        # Set title for the first image only across all categories
        if i == 0:
            if title_prefix == 'FP':
                # Custom title for FP and TN
                title = f"Predicted: Generated Image\nTrue: Generated Image"
            else:
                title = f"Predicted: Natural Image\nTrue: Generated Image"
            
            axes[i].set_title(title, fontsize=14, loc='left')  # Left align title
            title_set = True  # Mark the title as set
        
        axes[i].axis('off')
    
    # Hide unused subplots
    for ax in axes[len(samples):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)

# Visualize and save FP and TN images separately
visualize_and_save(fp_samples, 'FP', './result_img/FP_images_imagen4.png', rows=2, cols=4)
visualize_and_save(tn_samples, 'TN', './result_img/TN_images_imagen4.png', rows=2, cols=4)