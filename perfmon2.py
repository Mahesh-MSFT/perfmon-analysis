import os
import pandas as pd
import numpy as np

from modules.calculate_statistics import calculate_statistics
from modules.convert_blg_to_csv import convert_blg_to_csv
from modules.ensure_consistent_structure import ensure_consistent_structure
from modules.find_steepest_fall import find_steepest_fall
from modules.excel_creator import excel_creator
from modules.file_processor import file_processor

# start
log_directory = r'C:\Users\maksh\OneDrive - Microsoft\Documents\AVS\PerfTest\ParallelTesting'
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
    # Convert .blg files to .csv files
    convert_blg_to_csv(log_directory)

    # Process the CSV files
    all_statistics_df = file_processor(log_directory, metric_names, baseline_metric_name)

    # Write the combined statistics to an Excel file
    #excel_creator(all_statistics_df, log_directory)
