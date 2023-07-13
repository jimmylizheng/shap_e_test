import statistics
def mean_calc(filename):
    # Open the file
    tmp=filename+".txt"
    with open(tmp, 'r') as file:
        lines = file.readlines()  # Read the lines of the file

    # Convert lines to floats and calculate the mean
    numbers = [float(line.strip()) for line in lines]
    mean = sum(numbers) / len(numbers)
    std_dev = statistics.stdev(numbers)

    # Print the mean value
    print(f"Mean loading latency for {filename}: {mean} seconds")
    print(f"SDV loading latency for {filename}: {std_dev} seconds")
    
# file_list=['load-image300M-decoder','load-text300M-decoder','load-image300M-transmitter','load-text300M-transmitter','load-decoder','load-transmitter','load-image300M','load-text300M']
file_list=['load-decoder','load-transmitter','load-image300M','load-text300M']
for file_nm in file_list:
    mean_calc(file_nm)