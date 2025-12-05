import os
import json
import argparse

def generate_video_metadata(root_dir, output_json_path, video_type="video"):
    video_keys = ["input", "target", "render", "mask", "relit_render"]
    all_entries = []

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        entry = {}
        for key in video_keys:
            video_file = os.path.join(folder_path, f"{key}.mp4")
            if os.path.exists(video_file):
                entry[f"{key}_path"] = os.path.abspath(video_file)
            else:
                print(f"❌ Missing: {video_file}")
                break
        else:
            prompt_path = os.path.join(folder_path, "prompt.txt")
            if not os.path.exists(prompt_path):
                print(f"❌ Missing: {prompt_path}")
                continue

            with open(prompt_path, "r") as pf:
                entry["text"] = pf.read().strip()

            id_path = os.path.join(folder_path, "id.txt")
            if os.path.exists(id_path):
                with open(id_path, "r") as idf:
                    try:
                        entry["ref_frame_id"] = int(idf.read().strip())
                    except:
                        continue
            else:
                print(f"❌ Missing: {id_path}")
                continue
        
            ref_light_path = os.path.join(folder_path, "ref_light.png")
            if os.path.exists(ref_light_path):
                entry["ref_light"] = os.path.abspath(ref_light_path)
            else:
                entry["ref_light"] = None
            
            exr_file = os.path.join(folder_path, f"{folder}.exr")
            if os.path.exists(exr_file):
                entry["exr_path"] = os.path.abspath(exr_file)
            else:
                entry["exr_path"] = None

            entry["type"] = video_type
            all_entries.append(entry)

    with open(output_json_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"✅ Saved {len(all_entries)} entries to: {os.path.abspath(output_json_path)}")


def parse():
    """
    Parse command line arguments and generate video metadata.
    """
    parser = argparse.ArgumentParser(
        description="Generate video metadata JSON from directory structure"
    )
    parser.add_argument(
        "--root_dir",
        "-r",
        type=str,
        required=True,
        help="Root directory containing video folders"
    )
    parser.add_argument(
        "--output_json_path",
        "-o",
        type=str,
        default=None,
        help="Path where the JSON file will be saved (default: root_dir/meta.json)"
    )
    args = parser.parse_args()
    
    if args.output_json_path is None:
        args.output_json_path = os.path.join(args.root_dir, "metadata.json")
    
    generate_video_metadata(args.root_dir, args.output_json_path)

if __name__ == "__main__":
    parse()