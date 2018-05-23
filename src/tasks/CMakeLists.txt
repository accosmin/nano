list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/task_cifar100.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/task_cifar10.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/task_mnist.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/task_svhn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/task_affine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/task_peak2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/task_parity.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/task_mem_csv.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
