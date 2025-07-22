import os
import sys
import pandas as pd
import gc

from cpuonly.modules.file_processor import file_processor
from cpuonly.modules.convert_blg_to_csv import convert_blg_to_csv
from shared.modules.excel_creator import excel_creator

log_directory = r'C:\PATH\TO\BLGs'
metric_names = ['Request Execution Time',
         '# of Exceps Thrown', 
         '# of current logical Threads',
         '# of current physical Threads',
         'Contention Rate / sec',
         'Current Queue Length',
         'Queue Length Peak',
         '% Time in GC',
         'NumberOfActiveConnectionPools',
         'NumberOfActiveConnections',
         'NumberOfPooledConnections',
         'Total Application Pool Recycles',
         '% Managed Processor Time (estimated)',
         'Managed Memory Used (estimated)',
         'Request Wait Time',
         'Requests Failed',
         'Requests/Sec',
         'ArrivalRate',
         'CurrentQueueSize',
         'Network Adapter(vmxnet3 Ethernet Adapter _2)\Bytes Total/sec',
         'Network Adapter(vmxnet3 Ethernet Adapter _2)\Current Bandwidth',
         'Network Adapter(vmxnet3 Ethernet Adapter _2)\TCP RSC Coalesced Packets/sec',
         'NUMA Node Memory(0)\Available MBytes',
         'NUMA Node Memory(1)\Available MBytes',
         '(_Total)\% Processor Time']
baseline_metric_name = 'ASP.NET Applications(__Total__)\Request Execution Time'

if __name__ == '__main__':
    # Log Start Date and Time
    start_time = pd.Timestamp.now()
    print("Script started at:", start_time)
    
    # Convert .blg files to .csv files
    convert_blg_to_csv(log_directory)

    # Process the CSV files
    all_statistics_df = file_processor(log_directory, metric_names, baseline_metric_name)

    # Write the combined statistics to an Excel file
    excel_creator(all_statistics_df, log_directory)
    
    # Explicitly clear large DataFrame from memory
    del all_statistics_df
    
    # Force garbage collection to free up memory
    gc.collect()

    end_time = pd.Timestamp.now()
    elapsed_time = end_time - start_time
    
    # Format elapsed time in minutes and seconds
    total_seconds = elapsed_time.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    
    print("Script completed at:", end_time)
    print(f"Total elapsed time: {minutes} minutes and {seconds:.2f} seconds")
