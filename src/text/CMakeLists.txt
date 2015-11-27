# build library
set(libtext_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/align.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/concatenate.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ends_with.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/enum_string.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/equals.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/from_params.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/from_string.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lower.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/replace.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/split.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/starts_with.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/to_string.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/upper.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/table.h
        ${CMAKE_CURRENT_SOURCE_DIR}/table.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/table_header.h
        ${CMAKE_CURRENT_SOURCE_DIR}/table_row.h
        ${CMAKE_CURRENT_SOURCE_DIR}/table_row_comp.h
        ${CMAKE_CURRENT_SOURCE_DIR}/table_row_mark.h)

add_library(text SHARED
        ${libtext_sources})

# install library
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")


                
