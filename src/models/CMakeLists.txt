list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/wlearner_stump.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/wlearner_linear.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gboost_stump.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
