# setup CMake to run tests
enable_testing()

# function to create a unit test application
function(build_test_app test_app libs)
        get_filename_component(test_name ${test_app} NAME_WE)

        add_executable(${test_name} ${test_app})
        target_link_libraries(${test_name} ${libs})

        add_test(${test_name} ${test_name})
endfunction()

# build unit tests
build_test_app(test_average.cpp "")
build_test_app(test_batch.cpp "")
build_test_app(test_conv2d.cpp "")
build_test_app(test_corr2d.cpp "")
build_test_app(test_cubic.cpp "")
build_test_app(test_functions.cpp "")
build_test_app(test_gauss.cpp "")
build_test_app(test_momentum.cpp "")
build_test_app(test_quadratic.cpp "")
build_test_app(test_random.cpp "")
build_test_app(test_stats.cpp "")
build_test_app(test_stoch.cpp "")
build_test_app(test_tensor.cpp "")
build_test_app(test_tune.cpp "")

build_test_app(test_text.cpp text)
build_test_app(test_cmdline.cpp text)
build_test_app(test_filesystem.cpp text)

build_test_app(test_io.cpp io)

build_test_app(test_thread_loop.cpp thread)
build_test_app(test_thread_pool.cpp thread)

build_test_app(test_geom.cpp vision)
build_test_app(test_image.cpp vision)

build_test_app(test_accumulator.cpp cortex)
build_test_app(test_criteria.cpp cortex)
build_test_app(test_layers.cpp cortex)
build_test_app(test_loss.cpp cortex)
build_test_app(test_manager.cpp cortex)
build_test_app(test_model.cpp cortex)
build_test_app(test_charset.cpp cortex)
build_test_app(test_svhn.cpp cortex)
build_test_app(test_stl10.cpp cortex)
build_test_app(test_mnist.cpp cortex)
build_test_app(test_cifar10.cpp cortex)
build_test_app(test_cifar100.cpp cortex)
