list(APPEND liboptim_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/3hump_camel.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/beale.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/bohachevsky.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/booth.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cauchy.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/colville.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/dixon_price.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/goldstein_price.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/himmelblau.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/function.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matyas.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mccormick.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/powell.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rosenbrock.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rotated_ellipsoid.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sphere.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/styblinski_tang.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sum_squares.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trid.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/zakharov.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/util.hpp)

set(liboptim_sources "${liboptim_sources}" PARENT_SCOPE)


