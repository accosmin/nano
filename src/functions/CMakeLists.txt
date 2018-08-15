list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/axis_ellipsoid.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cauchy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/chung_reynolds.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/dixon_price.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/exponential.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/powell.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/qing.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/quadratic.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rosenbrock.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rotated_ellipsoid.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sargan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/schumer_steiglitz.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sphere.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/styblinski_tang.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trid.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/zakharov.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
