list(APPEND libnano_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/class.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/task_util.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/accumulator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trainer_state.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trainer_result.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/affine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/activation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/convolution.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/models/forward_network.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/iterators/iterator_warp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/iterators/iterator_noise.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trainers/trainer_batch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trainers/trainer_stoch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/synth_bitstream_vera_sans_mono_bold.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/synth_bitstream_vera_sans_mono.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/synth_dejavu_sans_mono_bold.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/synth_dejavu_sans_mono.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/synth_droid_sans_mono.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/synth_liberation_mono_bold.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/synth_liberation_mono.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/synth_nimbus_mono_bold.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/synth_nimbus_mono.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/synth_oxygen_mono.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/task_charset.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/task_cifar100.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/task_cifar10.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/task_mnist.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/task_stl10.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/task_svhn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/task_iris.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/task_wine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tasks/task_mem_csv.cpp)

set(libnano_sources "${libnano_sources}" PARENT_SCOPE)
