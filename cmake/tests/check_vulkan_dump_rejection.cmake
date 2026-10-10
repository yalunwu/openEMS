execute_process(
  COMMAND "${OPENEMS_EXECUTABLE}" "--vulkan-${DUMP_DOMAIN}=invalid"
  RESULT_VARIABLE exit_code OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr TIMEOUT 30
)
if(NOT "${exit_code}" STREQUAL "1" OR NOT stderr MATCHES "vulkan-${DUMP_DOMAIN} must be auto, cpu or gpu")
  message(FATAL_ERROR "Expected dump mode rejection, got ${exit_code}.\n${stdout}${stderr}")
endif()
