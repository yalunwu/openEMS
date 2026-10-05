# Run only when a Vulkan device and the Khronos validation layer are available.
if(NOT EXISTS "${BACKEND_EXECUTABLE}")
  message(FATAL_ERROR "BACKEND_EXECUTABLE must name the newly built test_backend")
endif()
if(NOT DEFINED TEST_ARGUMENT)
  set(TEST_ARGUMENT --synchronization-tests)
endif()
# The optional shader heuristic cannot distinguish disjoint state ranges or
# runtime branches in the Lorentz pre/apply shader. See the Phase 6a audit.
if(NOT DEFINED SHADER_HEURISTIC)
  set(SHADER_HEURISTIC 0)
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env
  VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
  VK_LAYER_VALIDATE_SYNC=1
  VK_LAYER_SYNCVAL_FULL_VALIDATION=1
  VK_LAYER_SYNCVAL_SHADER_ACCESSES_HEURISTIC=${SHADER_HEURISTIC}
  VK_LAYER_ENABLE_MESSAGE_LIMIT=0
  VK_LAYER_DEBUG_ACTION=VK_DBG_LAYER_ACTION_LOG_MSG
  VK_LOADER_DEBUG=layer
  "${BACKEND_EXECUTABLE}" ${TEST_ARGUMENT}
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors)
set(log "${output}\n${errors}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/vulkan-synchronization.log" "${log}")
if(NOT log MATCHES "Insert instance layer[^\n]*VK_LAYER_KHRONOS_validation")
  message(FATAL_ERROR "Khronos validation layer was not loaded; set VK_LAYER_PATH if needed. See vulkan-synchronization.log")
endif()
if(NOT result EQUAL 0 OR log MATCHES "Validation Error|Validation Warning|SYNC-HAZARD|VUID-")
  message(FATAL_ERROR "Vulkan synchronization/numerical checks failed (${result}). See vulkan-synchronization.log")
endif()
message(STATUS "Vulkan synchronization and numerical checks passed with Khronos validation loaded")
