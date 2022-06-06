describe_columns =[
 'clocks_current_graphicsMHz_mean',
 'clocks_current_graphicsMHz_std',
 'clocks_current_graphicsMHz_min',
 'clocks_current_graphicsMHz_25',
 'clocks_current_graphicsMHz_50',
 'clocks_current_graphicsMHz_75',
 'clocks_current_graphicsMHz_max',
 'clocks_current_smMHz_mean',
 'clocks_current_smMHz_std',
 'clocks_current_smMHz_min',
 'clocks_current_smMHz_25',
 'clocks_current_smMHz_50',
 'clocks_current_smMHz_75',
 'clocks_current_smMHz_max',
 'memory_freeMiB_mean',
 'memory_freeMiB_std',
 'memory_freeMiB_min',
 'memory_freeMiB_25',
 'memory_freeMiB_50',
 'memory_freeMiB_75',
 'memory_freeMiB_max',
 'memory_totalMiB_mean',
 'memory_totalMiB_std',
 'memory_totalMiB_min',
 'memory_totalMiB_25',
 'memory_totalMiB_50',
 'memory_totalMiB_75',
 'memory_totalMiB_max',
 'memory_usedMiB_mean',
 'memory_usedMiB_std',
 'memory_usedMiB_min',
 'memory_usedMiB_25',
 'memory_usedMiB_50',
 'memory_usedMiB_75',
 'memory_usedMiB_max',
 'power_drawW_mean',
 'power_drawW_std',
 'power_drawW_min',
 'power_drawW_25',
 'power_drawW_50',
 'power_drawW_75',
 'power_drawW_max',
 'temperature_gpu_mean',
 'temperature_gpu_std',
 'temperature_gpu_min',
 'temperature_gpu_25',
 'temperature_gpu_50',
 'temperature_gpu_75',
 'temperature_gpu_max',
 'utilization_gpu_mean',
 'utilization_gpu_std',
 'utilization_gpu_min',
 'utilization_gpu_25',
 'utilization_gpu_50',
 'utilization_gpu_75',
 'utilization_gpu_max',
 'utilization_memory_mean',
 'utilization_memory_std',
 'utilization_memory_min',
 'utilization_memory_25',
 'utilization_memory_50',
 'utilization_memory_75',
 'utilization_memory_max']
save_fig = ['memory_freeMiB',  #변하는 매트릭값
'memory_usedMiB',
'power_drawW',
'temperature_gpu',
'utilization_gpu',
'utilization_memory',
'clocks_current_graphicsMHz',
'clocks_current_smMHz',
'clocks_throttle_reasons_active',
'clocks_throttle_reasons_sw_power_cap'
]
not_move_value_columns = ['accounting_buffer_size',
 'accounting_mode',
 'clocks_applications_graphicsMHz',
 'clocks_applications_memoryMHz',
 'clocks_current_memoryMHz',
 'clocks_default_applications_graphicsMHz',
 'clocks_default_applications_memoryMHz',
 'clocks_max_graphicsMHz',
 'clocks_max_memoryMHz',
 'clocks_max_smMHz',
 'clocks_throttle_reasons_applications_clocks_setting',
 'clocks_throttle_reasons_gpu_idle',
 'clocks_throttle_reasons_hw_slowdown',
 'clocks_throttle_reasons_supported',
 'compute_mode',
 'display_active',
 'display_mode',
 'driver_model_current',
 'driver_model_pending',
 'driver_version',
 'fan_speed',
 'gom_current',
 'gom_pending',
 'index',
 'inforom_ecc',
 'inforom_img',
 'inforom_oem',
 'inforom_pwr',
 'memory_totalMiB',
 'name',
 'new_timestamp_gpu',
 'pci_bus',
 'pci_bus_id',
 'pci_device',
 'pci_device_id',
 'pci_domain',
 'pci_sub_device_id',
 'pcie_link_gen_current',
 'pcie_link_gen_max',
 'pcie_link_width_current',
 'pcie_link_width_max',
 'persistence_mode',
 'power_default_limitW',
 'power_limitW',
 'power_management',
 'power_max_limitW',
 'power_min_limitW',
 'pstate',
 'serial',
 'uuid',
 'vbios_version']
