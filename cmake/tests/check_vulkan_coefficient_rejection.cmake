execute_process(
  COMMAND "${OPENEMS_EXECUTABLE}" "--vulkan-coefficients=invalid"
  RESULT_VARIABLE exit_code
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
  TIMEOUT 30
)
if(NOT "${exit_code}" STREQUAL "1")
  message(FATAL_ERROR "Expected coefficient mode rejection, got ${exit_code}.\n${stdout}${stderr}")
endif()
string(FIND "${stderr}" "openEMS - vulkan-coefficients must be dense, palette or analyze" diagnostic_pos)
if(diagnostic_pos EQUAL -1)
  message(FATAL_ERROR "Missing coefficient validation diagnostic.\n${stdout}${stderr}")
endif()
