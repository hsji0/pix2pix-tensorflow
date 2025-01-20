import os
from PIL import Image

"""
503 : GRAY Defect
519 : SL Defect

"""

# Directories and paths - adjust these as needed
smt_folder = r"D:\9.Pairset Color Transfer\temptest"        # Folder containing all SMT and TIFF files
TARGET_GRAY_FOLDER = r"D:\9.Pairset Color Transfer\temptest\Gray" # Save folder for extracted gray images
TARGET_COLOR_FOLDER = r"D:\9.Pairset Color Transfer\temptest\Color" # Save folder for extracted color images


def filter_503_dict(filepath):
    """
    Read an .smt file, parse the DefectList section, and return a dictionary
    mapping normalized defect lines (without DEFECTID and IMAGELIST) to the original line,
    but only for lines where CLASSNUMBER equals 503.
    """
    in_defectlist = False
    norm_dict = {}

    with open(filepath, 'r') as file:
        for line in file:
            if 'DefectList' in line:
                in_defectlist = True
                continue

            if in_defectlist:
                stripped = line.strip()
                if (stripped.startswith('SummarySpec') or
                    stripped.startswith('SummaryList') or
                    stripped.startswith('EndOfFile')):
                    break

                tokens = stripped.split()
                if len(tokens) > 9 and tokens[9] == '503':
                    normalized = ' '.join(tokens[1:-1])
                    norm_dict[normalized] = stripped

    return norm_dict

def get_ordered_common_lines(file_before, file_after):
    # Obtain dictionaries of normalized 503 defect lines for both files
    dict_file1 = filter_503_dict(file_before)
    dict_file2 = filter_503_dict(file_after)

    # Find common normalized keys between the two dictionaries
    common_keys = set(dict_file1.keys()).intersection(set(dict_file2.keys()))

    # Re-read the "before" file to preserve original order for common defect lines
    ordered_common_lines = []
    in_defectlist = False

    with open(file_before, 'r') as file:
        for line in file:
            if 'DefectList' in line:
                in_defectlist = True
                continue

            if in_defectlist:
                stripped = line.strip()
                if (stripped.startswith('SummarySpec') or
                    stripped.startswith('SummaryList') or
                    stripped.startswith('EndOfFile')):
                    break

                tokens = stripped.split()
                if len(tokens) > 9 and tokens[9] == '503':
                    normalized = ' '.join(tokens[1:-1])
                    if normalized in common_keys:
                        ordered_common_lines.append(stripped)

    return ordered_common_lines

def extract_defect_ids(common_lines):
    defect_ids = []
    for line in common_lines:
        tokens = line.split()
        try:
            defect_ids.append(int(tokens[0]))
        except ValueError:
            continue
    # Remove duplicates while preserving order
    seen = set()
    unique_ids = []
    for did in defect_ids:
        if did not in seen:
            seen.add(did)
            unique_ids.append(did)
    return unique_ids

def process_file_set(core_name, file_before, file_after):
    # Get ordered common lines and defect IDs
    common_lines = get_ordered_common_lines(file_before, file_after)
    defect_ids = extract_defect_ids(common_lines)

    # Construct expected TIFF file paths based on core_name
    gray_tif = os.path.join(smt_folder, f"{core_name}_Gray.tif")
    color_tif = os.path.join(smt_folder, f"{core_name}.tif")

    # Create save directories if they do not exist
    os.makedirs(TARGET_GRAY_FOLDER, exist_ok=True)
    os.makedirs(TARGET_COLOR_FOLDER, exist_ok=True)

    # Process Gray TIFF (before-filtered images)
    try:
        gray_im = Image.open(gray_tif)
    except Exception as e:
        print(f"Error opening Gray TIFF {gray_tif}: {e}")
        gray_im = None

    if gray_im:
        for did in defect_ids:
            try:
                gray_im.seek(did)
                output_filename = f"{core_name}_Gray_{did}.png"
                output_path = os.path.join(TARGET_GRAY_FOLDER, output_filename)
                gray_im.save(output_path)
                print(f"Saved Gray: {output_path}")
            except EOFError:
                print(f"Frame {did} not found in {gray_tif}")
            except Exception as e:
                print(f"Error processing Gray frame {did}: {e}")

    # Process Color TIFF (final images)
    try:
        color_im = Image.open(color_tif)
    except Exception as e:
        print(f"Error opening Color TIFF {color_tif}: {e}")
        color_im = None

    if color_im:
        for did in defect_ids:
            try:
                color_im.seek(did)
                output_filename = f"{core_name}_{did}.png"
                output_path = os.path.join(TARGET_COLOR_FOLDER, output_filename)
                color_im.save(output_path)
                print(f"Saved Color: {output_path}")
            except EOFError:
                print(f"Frame {did} not found in {color_tif}")
            except Exception as e:
                print(f"Error processing Color frame {did}: {e}")


if __name__ == "__main__":

    # Group SMT files by their core basename (ignoring suffixes like " (2)", "복사본", etc.)
    smt_files = [f for f in os.listdir(smt_folder) if f.lower().endswith('.smt')]
    groups = {}

    def normalize_basename(filename):
        # Remove common suffix patterns like " (2)", " 복사본" and extension
        base = os.path.splitext(filename)[0]
        for pattern in [" (2)", " 복사본"]:
            base = base.replace(pattern, "")
        return base

    for f in smt_files:
        core = normalize_basename(f)
        groups.setdefault(core, []).append(f)

    # For each group that has at least two SMT files, assume we have a before and after pair.
    for core_name, files in groups.items():
        if len(files) < 2:
            continue  # Skip if we don't have both before and after

        # Sort files for consistency; assume first is before, second is after
        files.sort()
        file_before = os.path.join(smt_folder, files[0])
        file_after = os.path.join(smt_folder, files[1])

        print(f"Processing set: {core_name}")
        process_file_set(core_name, file_before, file_after)
