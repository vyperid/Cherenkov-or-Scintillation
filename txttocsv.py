import csv
import sys

if len(sys.argv) != 3:
    print("Usage: python your_script.py <input_file> <output_file>")
    exit(1)

txt_file_read = sys.argv[1]
txt_file_write = sys.argv[2]

with open(txt_file_read, 'r') as read_file, open(txt_file_write, 'w', newline='') as write_file:
    writer = csv.writer(write_file)

    column_names = ["PMTN", "Time", "Vector", "Energy"]
    writer.writerow(column_names)  # Add column names

    data = []
    count = 0

    while True:
        count += 1
        line = read_file.readline()

        if not line:
            break

        sections = line.split(',')[1:]  # Remove the first element
        pmtno, time, vector, energy = [sec.split(";") for sec in sections]
        data.extend(zip(pmtno, time, vector, energy))

    writer.writerows(data)  # Write all processed data at once

print("Total lines processed:", count)
    