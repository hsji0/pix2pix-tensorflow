def filter_503(filepath):
    """
    Read an .smt file, parse the DefectList section, and return a set of lines
    where CLASSNUMBER equals 503.
    """
    in_defectlist = False
    result_lines = set()

    with open(filepath, 'r') as file:
        for line in file:
            # Entering the DefectList section
            if 'DefectList' in line:
                in_defectlist = True
                continue

            # Exit DefectList when reaching SummarySpec or SummaryList or end markers
            if in_defectlist:
                stripped = line.strip()
                # Break out of the loop if we hit another section or EndOfFile
                if stripped.startswith('SummarySpec') or stripped.startswith('SummaryList') or stripped.startswith(
                        'EndOfFile'):
                    break

                # Process potential defect lines
                # Ensure the line has enough tokens and check the CLASSNUMBER position (10th token, index 9)
                tokens = stripped.split()
                if len(tokens) > 9 and tokens[9] == '503':
                    result_lines.add(stripped)

    return result_lines


# Paths to the two .smt files you want to compare
file1_path = r"path\to\first_file.smt"
file2_path = r"path\to\second_file.smt"

# Extract 503-class defect lines from both files
lines_file1 = filter_503(file1_path)
lines_file2 = filter_503(file2_path)

# Find the intersection of lines from both files
common_lines = lines_file1.intersection(lines_file2)

# Convert the set to a list for further processing or display
common_lines_list = list(common_lines)

# Print or use the result as needed
for line in common_lines_list:
    print(line)

# If you need the result as a list of lines:
print(common_lines_list)


