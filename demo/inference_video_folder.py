import os
import sys
import glob
import argparse
import traceback
import pandas as pd

from inference_single_video import get_pose2D, get_pose3D, img2video

def main():
    parser = argparse.ArgumentParser(description="Batch process a folder of videos for 3D pose estimation and aggregate results.")
    parser.add_argument('--video-dir', type=str, default='Test_EWalk_8_chunks', 
                        help='Name of the video subdirectory under ./demo/video/')
    parser.add_argument('--gpu', type=str, default='0', 
                        help='GPU ID to use')
    parser.add_argument('--generate-images', action='store_true', default=False, 
                        help='Generate visualization images (2D, 3D, combined poses and MP4 video)')
    parser.add_argument('--2d-only', action='store_true', default=False,
                        help='Run only 2D pose detection and save CGTGait-order 2D CSV. Skips 3D model inference.')
    parser.add_argument('--output-csv', type=str, default=None, 
                        help='Path to save the combined CSV file. If not specified, saves to ./demo/output/<video_dir>_combined.csv')
    args = parser.parse_args()

    # Set visible GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_folder_path = os.path.join('.', 'demo', 'video', args.video_dir)
    if not os.path.isdir(video_folder_path):
        print(f"Error: Video directory '{video_folder_path}' does not exist.")
        sys.exit(1)

    # Find all mp4 videos in the folder
    video_pattern = os.path.join(video_folder_path, '*.mp4')
    video_paths = sorted(glob.glob(video_pattern))

    if not video_paths:
        print(f"Error: No .mp4 files found in '{video_folder_path}'.")
        sys.exit(1)

    print(f"Found {len(video_paths)} videos to process in '{args.video_dir}'.")

    successful_samples = 0
    failed_samples = []
    all_dfs = []

    for i, video_path in enumerate(video_paths):
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        
        print("\n" + "="*80)
        print(f"Processing video [{i+1}/{len(video_paths)}]: {video_filename}")
        print("="*80)

        # Construct the relative video path as expected by the script (relative to ./demo/video/)
        # e.g., 'Test_EWalk_8_chunks/VID_RGB_003_chunk_1.mp4'
        video_arg = f"{args.video_dir}/{video_filename}"
        output_dir = f"./demo/output/{video_name}/"

        try:
            # 1. Run 2D keypoints detection (always)
            get_pose2D(video_path, output_dir)

            if getattr(args, '2d_only', False):
                # 2D-only mode: skip 3D inference, read CGTGait 2D CSV
                csv_path = os.path.join(output_dir, 'input_2D', 'keypoints_cgtgait_order.csv')
            else:
                # 2. Run 3D keypoints reconstruction
                get_pose3D(video_path, output_dir, generate_images=args.generate_images)

                # 3. Compile images to video if required
                if args.generate_images:
                    img2video(video_path, output_dir, video_name)

                # Read the generated 3D keypoint CSV
                csv_path = os.path.join(output_dir, 'output_3D', 'keypoints_cgtgait_order.csv')

            # 4. Read and collect the output CSV
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Insert the sample name at the beginning of the DataFrame
                df.insert(0, 'sample_name', video_name)
                all_dfs.append(df)
                successful_samples += 1
                print(f"Successfully processed and collected CSV for: {video_name}")
            else:
                print(f"Warning: Output CSV not found at {csv_path}")
                failed_samples.append((video_name, "Output CSV not found"))

        except Exception as e:
            print(f"Error processing video {video_name}: {str(e)}")
            traceback.print_exc()
            failed_samples.append((video_name, str(e)))

    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print(f"Successfully processed: {successful_samples}/{len(video_paths)}")
    if failed_samples:
        print(f"Failed samples ({len(failed_samples)}):")
        for name, err in failed_samples:
            print(f"  - {name}: {err}")
    print("="*80)

    # Combine all DataFrames if any succeeded
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Determine output CSV path
        if args.output_csv is None:
            suffix = '2D' if getattr(args, '2d_only', False) else '3D'
            output_csv_path = os.path.join('.', 'demo', 'output', f"{args.video_dir}_combined_{suffix}.csv")
        else:
            output_csv_path = args.output_csv

        # Ensure target directory exists
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Save to file
        combined_df.to_csv(output_csv_path, index=False)
        print(f"\nAll keypoints successfully combined and saved to: {output_csv_path}")
        print(f"Combined CSV dimensions: {combined_df.shape} (rows, columns)")
    else:
        print("\nNo samples were successfully processed. No combined CSV generated.")

if __name__ == "__main__":
    main()
