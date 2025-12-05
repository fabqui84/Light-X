import os
import torch
import argparse
import numpy as np
from PIL import Image
from ultralytics.models.sam import SAM2VideoPredictor
import imageio
from tqdm import tqdm
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_first_frame(video_path, output_path):
    """Extract the first frame from a video and save it as an image"""
    try:
        reader = imageio.get_reader(video_path)
        first_frame = reader.get_data(0)  # Get first frame
        reader.close()
        
        # Save as PNG using PIL
        Image.fromarray(first_frame).save(output_path)
        logger.info(f"First frame saved to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error extracting first frame from {video_path}: {str(e)}")
        return False

def copy_input_video(input_video_path, output_dir):
    """Copy input.mp4 to output directory"""
    try:
        base_name = os.path.splitext(os.path.basename(input_video_path))[0]
        output_video_path = os.path.join(output_dir, f"{base_name}.mp4")
        shutil.copy2(input_video_path, output_video_path)
        logger.info(f"Input video copied to: {output_video_path}")
        return True
    except Exception as e:
        logger.error(f"Error copying input video: {str(e)}")
        return False

def process_video(predictor, input_video_path, output_dir, x, y, width, height, fps):
    """Process a single video and generate foreground and mask outputs"""
    try:
        logger.info(f"Processing video: {input_video_path}")
        
        # Extract and save first frame of input video
        base_name = os.path.splitext(os.path.basename(input_video_path))[0]
        input_first_frame_path = os.path.join(output_dir, f"{base_name}_first_frame.png")
        extract_first_frame(input_video_path, input_first_frame_path)
        
        results = predictor(
            source=input_video_path,
            points=[x, y],
            labels=[1],
            stream=True  # Use stream mode for better memory efficiency
        )
        
        # Define output paths
        foreground_output_path = os.path.join(output_dir, f"{base_name}_foreground.mp4")
        mask_output_path = os.path.join(output_dir, f"{base_name}_mask.mp4")
        
        # Initialize video writers with RGB format
        foreground_writer = imageio.get_writer(foreground_output_path, fps=fps, macro_block_size=None)
        mask_writer = imageio.get_writer(mask_output_path, fps=fps, macro_block_size=None)
        
        logger.info(f"Saving outputs for {base_name} to: {output_dir}")
        
        # Process each frame
        frame_count = 0
        
        for i, result in enumerate(tqdm(results, desc=f"Processing {base_name} frames")):
            if result.masks is not None:
                # Get mask
                mask = result.masks.data.squeeze().cpu().numpy()
                
                # Convert mask to uint8 (0-255) and create 3-channel mask
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_rgb = np.stack([mask_uint8] * 3, axis=-1)
                
                # Get original frame
                original_frame = result.orig_img
                if original_frame is None:
                    logger.warning(f"Original frame {i} is None")
                    continue

                original_rgb = original_frame[:, :, ::-1] # Convert BGR to RGB
               
                # Create foreground (apply mask to original frame)
                mask_bool = mask > 0.5  # Threshold for foreground
                foreground = original_rgb.copy()
                foreground[~mask_bool] = 0  # Set background to black
                
                # Resize outputs to match original dimensions if needed
                if mask_rgb.shape[:2] != (height, width):
                    # Use PIL for resizing to maintain quality
                    mask_pil = Image.fromarray(mask_rgb).resize((width, height), Image.Resampling.LANCZOS)
                    mask_rgb = np.array(mask_pil)
                
                if foreground.shape[:2] != (height, width):
                    foreground_pil = Image.fromarray(foreground).resize((width, height), Image.Resampling.LANCZOS)
                    foreground = np.array(foreground_pil)
                
                # Write frames to videos
                foreground_writer.append_data(foreground.astype(np.uint8))
                mask_writer.append_data(mask_rgb.astype(np.uint8))
                
                frame_count += 1
            else:
                logger.warning(f"No mask found for frame {i}")
        
        # Close video writers
        foreground_writer.close()
        mask_writer.close()
        
        logger.info(f"Successfully processed {frame_count} frames for {base_name}")
        logger.info(f"Input first frame saved to: {input_first_frame_path}")
        logger.info(f"Foreground video saved to: {foreground_output_path}")
        logger.info(f"Mask video saved to: {mask_output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing video {input_video_path}: {str(e)}")
        return False

def get_video_properties(video_path):
    """Get video properties: width, height, fps"""
    try:
        reader = imageio.get_reader(video_path)
        meta_data = reader.get_meta_data()
        fps = meta_data.get('fps', 30.0)  # Default to 30 FPS if not available
        width, height = meta_data.get('size', (reader.get_data(0).shape[1], reader.get_data(0).shape[0]))
        reader.close()
        return width, height, fps
    except Exception as e:
        logger.error(f"Error reading video properties from {video_path}: {str(e)}")
        raise

def main(args):
    try:
        # Check if input video exists
        if not os.path.exists(args.input_video):
            raise FileNotFoundError(f"Input video not found: {args.input_video}")
        
        # Get video properties
        width, height, fps = get_video_properties(args.input_video)
        
        # Set default x and y to video center if not provided
        if args.x is None:
            args.x = width // 2
        if args.y is None:
            args.y = height // 2
        
        logger.info(f"Video properties: {width}x{height}, {fps:.2f} FPS")
        logger.info(f"Using point coordinates: x={args.x}, y={args.y}")
        
        # Create output directory as 'seg' folder in the same directory as input video
        video_dir = os.path.dirname(args.input_video)
        if args.output_dir is None:
            # Default: create 'seg' folder in the same directory as input video
            output_dir = os.path.join(video_dir, "seg")
        else:
            output_dir = args.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Disable Ultralytics automatic saving to runs folder
        os.environ["YOLO_SAVE_RUNS"] = "0"
        
        # Configure predictor
        overrides = dict(
            conf=args.confidence,
            task="segment", 
            mode="predict", 
            imgsz=args.imgsz, 
            model=args.model_path,
            device=args.device,
            save=False,  # Disable automatic saving
            exist_ok=True,  # Overwrite existing files
            project=None,  # Don't use project folder
            name=None  # Don't create named folder in runs
        )
        
        logger.info(f"Initializing SAM2VideoPredictor with model: {args.model_path}")
        predictor = SAM2VideoPredictor(overrides=overrides)
        
        # Process video to generate foreground and mask
        success = process_video(
            predictor, 
            args.input_video, 
            output_dir, 
            args.x, 
            args.y, 
            width, 
            height, 
            fps
        )
        
        if not success:
            raise Exception("Failed to process video")
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and generate foreground and mask using SAM2VideoPredictor.")
    
    # Input/Output arguments
    parser.add_argument("--input_video", type=str, required=True,
                       help="Path to input video file.")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save output videos. Default: 'seg' folder in the same directory as input video.")
    
    # Point coordinates (optional, defaults to video center)
    parser.add_argument("--x", type=int, default=None, help="X coordinate of the point. Default: width/2")
    parser.add_argument("--y", type=int, default=None, help="Y coordinate of the point. Default: height/2")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default="sam2_b.pt", 
                       help="Path to SAM2 model weights.")
    parser.add_argument("--confidence", type=float, default=0.25, 
                       help="Confidence threshold for segmentation.")
    parser.add_argument("--imgsz", type=int, default=1024, 
                       help="Image size for processing.")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use for processing (cuda/cpu).")
    
    args = parser.parse_args()
    main(args)
