# 固定源码快照；只构建核心 helper，不运行上游的依赖下载或全局配置。
set(SMALL_GICP_COMMIT fd29d8cf94cf05ed7ad21c81c27b65963110adb5)
set(SMALL_GICP_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../third_party/small_gicp")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/../tools/verify_vendor.py"
  "${SMALL_GICP_SOURCE_DIR}" RESULT_VARIABLE vendor_status)
if(NOT vendor_status EQUAL 0)
  message(FATAL_ERROR "small_gicp 源码校验失败")
endif()
add_library(small_gicp_vendor STATIC
  "${SMALL_GICP_SOURCE_DIR}/src/small_gicp/registration/registration.cpp"
  "${SMALL_GICP_SOURCE_DIR}/src/small_gicp/registration/registration_helper.cpp")
target_include_directories(small_gicp_vendor SYSTEM PUBLIC "${SMALL_GICP_SOURCE_DIR}/include")
target_link_libraries(small_gicp_vendor PUBLIC Eigen3::Eigen)
if(MSVC)
  target_compile_definitions(small_gicp_vendor PRIVATE _USE_MATH_DEFINES)
  target_compile_options(small_gicp_vendor PRIVATE /bigobj)
endif()
