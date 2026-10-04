execute_process(
  COMMAND "${OPENEMS_EXECUTABLE}" "--vulkan-batch-size=${BATCH_SIZE}"
  RESULT_VARIABLE exit_code
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
  TIMEOUT 30
)

if(NOT "${exit_code}" STREQUAL "1")
  message(FATAL_ERROR
    "Expected exit code 1 for batch size ${BATCH_SIZE}, got ${exit_code}.\n${stdout}${stderr}")
endif()

string(FIND "${stderr}" "openEMS - vulkan-batch-size must be between 1 and 64" diagnostic_pos)
if(diagnostic_pos EQUAL -1)
  message(FATAL_ERROR
    "Missing batch-size validation diagnostic for ${BATCH_SIZE}.\n${stdout}${stderr}")
endif()
