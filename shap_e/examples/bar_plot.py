import matplotlib.pyplot as plt

# Sample data
x_values = ['text300M', 'image300M', 'Transmitter+nerf','Transmitter+stf', 'Decoder+nerf', 'Decoder+stf']
y_values = [24.75, 54.72, 8.79, 1.67, 8.70, 1.78]

# Create bar chart
plt.bar(x_values, y_values)

plt.subplots_adjust(bottom=0.3)  # Increase or decrease the value as needed

# Wrap labels into multiple lines
# plt.xticks(rotation=0, ha='center')
plt.xticks(range(len(x_values)), x_values, rotation=0, ha='center', fontsize=8)
# plt.tick_params(axis='x', pad=18)  # Increase or decrease the value as needed

# Customize the chart
# plt.title('Bar Chart Example')
# plt.xlabel('Categories')
# plt.ylabel('Values')

# Save the figure
plt.savefig('bar_chart.png')

# Display the chart
plt.show()
