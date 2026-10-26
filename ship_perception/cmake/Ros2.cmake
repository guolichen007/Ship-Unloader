# Optional packaging only. Existing standalone C++ algorithms remain unchanged.
find_package(ament_cmake QUIET)
if(ament_cmake_FOUND)
  find_package(ament_cmake_python REQUIRED)
  ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  install(PROGRAMS executables/ship_hatch_once executables/ship_hatch_pcd
    DESTINATION lib/${PROJECT_NAME})
  install(DIRECTORY config datasets DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")
  install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/software_identity.json"
    "{\"software_git_sha\": \"${M0_GIT_SHA}\"}\n")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/software_identity.json"
    DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")
  if(BUILD_TESTING)
    find_package(ament_cmake_pytest REQUIRED)
    ament_add_pytest_test(ros2_capture tests/test_ros2_capture.py TIMEOUT 90)
    ament_add_pytest_test(pointcloud2_codec tests/test_pointcloud2_codec.py TIMEOUT 90)
    foreach(contract hatch_hardening scene_context canonical_neighborhood io_sources hatch_precision pipeline_contract inner_edge_role final_line_geometry raw_surface_normals loaded_proposal_retention raw_face_refinement corner_export)
      ament_add_pytest_test(${contract} tests/test_${contract}.py TIMEOUT 180)
    endforeach()
    ament_add_pytest_test(hatch_metamorphic tests/test_hatch_metamorphic.py TIMEOUT 900)
  endif()
  ament_package()
endif()
