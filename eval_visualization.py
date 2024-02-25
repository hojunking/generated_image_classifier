# import pandas as pd
# from PIL import Image
# import matplotlib.pyplot as plt

# # Load the DataFrame
# df = pd.read_csv('test_results.csv')

# img_dir = './result_img/'

# # Visualization function
# def display_and_save_images(dfs, titles, filename):
#     num_images = sum(len(df) for df in dfs)  # Calculate the total number of images
#     if num_images > 8:
#         print("Warning: Trying to display more than 8 images. Only the first 8 will be displayed.")
#     fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # Adjust the size as needed
#     axs = axs.flatten()
    
#     image_count = 0
#     for df, title in zip(dfs, titles):
#         for _, row in df.iterrows():
#             if image_count >= 8:  # Prevent attempting to plot more than 8 images
#                 break
#             img = Image.open(row['path'])
#             axs[image_count].imshow(img)
#             #axs[image_count].set_title(f"Predicted: {row['pred']}, True: {row['label']}")
#             axs[image_count].axis('off')
#             image_count += 1
            
#     plt.suptitle(' & '.join(titles))
#     plt.savefig(f"{img_dir+filename}.png")  # Save the figure
#     plt.show()

# # Filter the DataFrame for each category
# tp = df[(df['pred'] == 1) & (df['label'] == 1)].head(4)
# tn = df[(df['pred'] == 0) & (df['label'] == 0)].head(4)
# fp = df[(df['pred'] == 1) & (df['label'] == 0)].head(4)
# fn = df[(df['pred'] == 0) & (df['label'] == 1)].head(4)

# # Visualize and save TP and TN
# display_and_save_images([tp, tn], 
#                         ['True Positives (Predicted: generated, True: generated)', 
#                          'True Negatives (Predicted: natural, True: natural)'], 
#                         'TP_TN')

# # Visualize and save FP and FN
# display_and_save_images([fp, fn], 
#                         ['False Positives (Predicted: generated, True: natural)', 
#                          'False Negatives (Predicted: natural, True: generated)'], 
#                         'FP_FN')


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
# Assuming df_test is your dataframe and has been defined earlier

df = pd.read_csv('./test_result/test_results.csv')
# Sample up to 8 images for TP, TN, FP, and FN
tp_samples = df[(df['label'] == 1) & (df['pred'] == 1)].sample(n=min(8, len(df[(df['label'] == 1) & (df['pred'] == 1)])), random_state=4)
tn_samples = df[(df['label'] == 0) & (df['pred'] == 0)].sample(n=min(8, len(df[(df['label'] == 0) & (df['pred'] == 0)])), random_state=4)
fp_samples = df[(df['label'] == 0) & (df['pred'] == 1)].sample(n=min(2, len(df[(df['label'] == 0) & (df['pred'] == 1)])), random_state=1)
fn_samples = df[(df['label'] == 1) & (df['pred'] == 0)].sample(n=min(2, len(df[(df['label'] == 1) & (df['pred'] == 0)])), random_state=1)

# Function to visualize and save images
def visualize_and_save(samples, title_prefix, filename, rows=2, cols=4, fig_size=(20, 10)):
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    # Flatten axes array if not already flat to ensure consistent indexing
    axes = np.array(axes).reshape(-1)
    for i, (idx, row) in enumerate(samples.iterrows()):
        img = mpimg.imread(row['path'])  # Use this line in your local environment
        axes[i].imshow(img)
        
        # Set title for the first image only
        if i == 0:  # Check if it's the first image
            if title_prefix == 'FP/FN':
                pred_label = 'Generated Image' if row['pred'] == 0 else 'Natural Image'
                true_label = 'Natural Image' if row['label'] == 1 else 'Generated Image'
                title = f"Predicted: {pred_label}\nTrue: {true_label}"
            else:
                if title_prefix == 'TP':
                    title = f"Predicted: Natural Image\nTrue: Natural Image"
                else:
                    title = f"Predicted: Generated Image\nTrue: Generated Image"
                #title = f"{title_prefix} Examples"
                #title = f"{title_prefix}: {row['id']}"
            axes[i].set_title(title, fontsize=14, loc='left')  # Left align title
        elif i == 2 and title_prefix == 'FP/FN':
            pred_label = 'Generated Image' if row['pred'] == 0 else 'Natural Image'
            true_label = 'Natural Image' if row['label'] == 1 else 'Generated Image'
            title = f"Predicted: {pred_label}\nTrue: {true_label}"
            axes[i].set_title(title, fontsize=14, loc='left')  # Left align title
            
        axes[i].axis('off')
    
    # Adjust the layout to make space for the title
    plt.subplots_adjust(top=0.60)  # You may need to adjust this value
    # Hide unused subplots
    for ax in axes[len(samples):]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')

# Visualize and save TP and TN images
visualize_and_save(tp_samples, 'TP', './result_img/TP_images4.png', rows=2, cols=4)
visualize_and_save(tn_samples, 'TN', './result_img/TN_images4.png', rows=2, cols=4)

# Combine FP and FN samples for integrated visualization
fp_fn_samples = pd.concat([fp_samples, fn_samples])

# Visualize and save integrated FP and FN images
visualize_and_save(fp_fn_samples, 'FP/FN', './result_img/FP_FN_integrated_images2.png', rows=1, cols=4, fig_size=(20, 5))
