def mean_calc(filename):
    # Open the file
    tmp=filename+".txt"
    with open(tmp, 'r') as file:
        lines = file.readlines()  # Read the lines of the file

    # Convert lines to floats and calculate the mean
    numbers = [float(line.strip()) for line in lines]
    mean = sum(numbers) / len(numbers)

    # Print the mean value
    print(f"Mean loading latency for {filename}: {mean} seconds")
    
file_list=['load-image300M-decoder','load-text300M-decoder','load-image300M-transmitter','load-text300M-transmitter']
for file_nm in file_list:
    mean_calc(file_nm)