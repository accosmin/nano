list(APPEND liboptim_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/3hump_camel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/axis_ellipsoid.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/beale.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/bohachevsky.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/booth.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cauchy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/chung_reynolds.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/colville.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/dixon_price.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/exponential.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/goldstein_price.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/himmelblau.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matyas.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mccormick.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/powell.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/qing.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rosenbrock.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rotated_ellipsoid.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sargan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/schumer_steiglitz.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sphere.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/styblinski_tang.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trid.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/zakharov.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/util.hpp)

set(liboptim_sources "${liboptim_sources}" PARENT_SCOPE)


