import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

def im_grid_shower(samples, rows, cols):
    """Display images in a grid with fixed figsize."""
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    axes = np.array(axes).reshape(rows, cols)  # Ensure 2D array of axes
    
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            ax.imshow(samples[i])
        ax.axis('off')
    
    plt.tight_layout()
    return fig, axes

def get_save_name(default_name='generated'):
    """Get custom name for files or use default."""
    name = input("Custom name for file? (if not, press Enter): ").strip()
    return name if name else default_name

def im_saver(samples, name=None, output_dir='output', format='png'):
    # Initialize counters first
    saved = 0
    failed = 0 

    # Create summary after initializing variables
    summary = {
        'saved': 0,
        'failed': 0,
        'total': len(samples)
    }  

    # Ask for saving preference
    saving = input(f'Would you like to save the images? ').lower()

    if saving.startswith('y'): 
        if name is None:
            name = get_save_name()
            print(f"Using name: {name}")

        os.makedirs(output_dir, exist_ok=True)
        
        for i, sample in enumerate(samples):
            try:
                img_array = (sample * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                filename = os.path.join(output_dir, f'{name}_{i:03d}.{format}')
                img.save(filename)
                print(f'Saved image {i+1}/{len(samples)}: {filename}')
                summary['saved'] += 1
                
            except Exception as e:
                print(f'Error saving image {i}: {str(e)}')
                summary['failed'] += 1
                continue
    
    return summary  # Always return the summary dictionary, even if no files were saved

def main():
    parser = argparse.ArgumentParser(description='Process and save image samples.')
    parser.add_argument('input', type=Path, help='Input .npz file containing samples')
    parser.add_argument('--rows', type=int, default=2, help='Number of rows in display grid')
    parser.add_argument('--cols', type=int, default=5, help='Number of columns in display grid')
    parser.add_argument('--no-show', action='store_true', help='Skip displaying the images')
    parser.add_argument('--save', action='store_true', help='Save the images')
    parser.add_argument('--name', type=str, help='Base name for saved files (optional)')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--format', type=str, default='png', help='Image format (png, jpg, etc.)')

    args = parser.parse_args()

    try:
        print(f"Loading file: {args.input}")
        data = np.load(args.input)
        samples = data['arr_0'] if 'arr_0' in data.files else data[data.files[0]]
        print(f"Loaded {len(samples)} samples")

        if not args.no_show:
            fig, axes = im_grid_shower(samples, args.rows, args.cols)
            plt.show()

        if args.save:
            results = im_saver(
                samples=samples,
                name=args.name,
                output_dir=args.output_dir,
                format=args.format
            )
            print("\nSaving Results:")
            print(f"Successfully saved: {results['saved']}")
            print(f"Failed to save: {results['failed']}")
            print(f"Total processed: {results['total']}")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()