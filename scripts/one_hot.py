import argparse
import os
import sys
from PIL import Image, ImageOps

def process_thermal_image(input_path, output_path):
    """
    Inverts the colors of a thermal image (White Hot <-> Black Hot).
    """
    # 1. Validate Input
    if not os.path.exists(input_path):
        print(f"Error: The input file '{input_path}' was not found.")
        sys.exit(1)

    try:
        # 2. Open the image
        with Image.open(input_path) as img:
            
            # Convert to RGB to ensure we can invert safely 
            # (Handling cases where input might be Grayscale 'L' or 'RGBA')
            if img.mode == 'RGBA':
                # Separate alpha channel to avoid inverting transparency
                r, g, b, a = img.split()
                rgb_img = Image.merge('RGB', (r, g, b))
                inverted_rgb = ImageOps.invert(rgb_img)
                r2, g2, b2 = inverted_rgb.split()
                # Merge back with original alpha
                final_img = Image.merge('RGBA', (r2, g2, b2, a))
            else:
                final_img = ImageOps.invert(img.convert('RGB'))

            # 3. Determine Output Filename
            # Check if output_path is a directory (ends with slash or is an existing dir)
            if output_path.endswith(os.sep) or os.path.isdir(output_path):
                # Make sure the directory exists
                os.makedirs(output_path, exist_ok=True)
                
                # Create a new filename (e.g., image_inverted.jpg)
                filename = os.path.basename(input_path)
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_inverted{ext}"
                save_path = os.path.join(output_path, new_filename)
            else:
                # User provided a full file path
                save_path = output_path
                # Ensure the parent directory exists
                parent_dir = os.path.dirname(save_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)

            # 4. Save
            final_img.save(save_path)
            print(f"Success! Image flipped and saved to: {save_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Setup Command Line Arguments
    parser = argparse.ArgumentParser(description="Flip Thermal Images (White Hot <-> Black Hot)")
    
    parser.add_argument(
        '--input', 
        required=True, 
        help="Path to the input image (JPG, PNG, etc.)"
    )
    parser.add_argument(
        '--output', 
        required=True, 
        help="Path to the output file or directory"
    )

    args = parser.parse_args()

    process_thermal_image(args.input, args.output)