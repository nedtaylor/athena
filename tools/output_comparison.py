import sys

def compare_files(file1, file2, tolerance):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    if len(lines1) != len(lines2):
        return False

    for line1, line2 in zip(lines1, lines2):
        if not compare_lines(line1, line2, tolerance):
            return False

    return True

def compare_lines(line1, line2, tolerance):
    if line1 == line2:
        return True
    
    fields1 = line1.split(', ')
    fields2 = line2.split(', ')

    # Check if the lines have the same number of fields
    if len(fields1) != len(fields2):
        return False

    for field1, field2 in zip(fields1, fields2):
        key1, value1 = field1.split('=')
        key2, value2 = field2.split('=')

        # Check if the keys match
        if key1 != key2:
            return False

        # Check if the values are numerically close
        if key1 in ['loss', 'accuracy']:
            if not is_close(float(value1), float(value2), tolerance):
                return False
        else:
            # For non-numeric values, check for exact match
            if value1 != value2:
                return False

    return True

def is_close(a, b, tolerance):
    return abs(a - b) <= tolerance

# Example usage
file1_path = sys.argv[1]
file2_path = sys.argv[2]
tolerance_value = 2e-3

if compare_files(file1_path, file2_path, tolerance_value):
    print("Files are almost the same within the specified tolerance.")
    sys.exit(0)
else:
    print("Files differ within the specified tolerance.")
    sys.exit(1)