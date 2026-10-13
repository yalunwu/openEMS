execute_process(
  COMMAND "${OPENEMS_EXECUTABLE}" "--vulkan-${DUMP_DOMAIN}=invalid"
  RESULT_VARIABLE exit_code OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr TIMEOUT 30
)
set(expected_modes "auto, cpu or gpu")
if(DUMP_DOMAIN STREQUAL "pml")
  set(expected_modes "indexed or fused")
endif()
if(NOT "${exit_code}" STREQUAL "1" OR NOT stderr MATCHES "vulkan-${DUMP_DOMAIN} must be ${expected_modes}")
  message(FATAL_ERROR "Expected dump mode rejection, got ${exit_code}.\n${stdout}${stderr}")
endif()
