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
build_test_app(test_cubic.cpp "")
build_test_app(test_gauss.cpp "")
build_test_app(test_momentum.cpp "")
build_test_app(test_quadratic.cpp "")
build_test_app(test_random.cpp "")
build_test_app(test_stats.cpp "")
build_test_app(test_tensor.cpp "")
build_test_app(test_tune.cpp "")

build_test_app(test_stoch.cpp nano)
build_test_app(test_batch.cpp nano)
build_test_app(test_functions.cpp nano)

build_test_app(test_text.cpp nano)
build_test_app(test_cmdline.cpp nano)
build_test_app(test_filesystem.cpp nano)

build_test_app(test_io.cpp nano)

build_test_app(test_thread_loop.cpp nano)
build_test_app(test_thread_pool.cpp nano)

build_test_app(test_geom.cpp nano)
build_test_app(test_image.cpp nano)

build_test_app(test_accumulator.cpp nano)
build_test_app(test_criteria.cpp nano)
build_test_app(test_layers.cpp nano)
build_test_app(test_loss.cpp nano)
build_test_app(test_manager.cpp nano)
build_test_app(test_model.cpp nano)
build_test_app(test_trainer.cpp nano)
build_test_app(test_conv3d.cpp nano)

build_test_app(test_task_charset.cpp nano)
build_test_app(test_task_svhn.cpp nano)
build_test_app(test_task_iris.cpp nano)
build_test_app(test_task_wine.cpp nano)
build_test_app(test_task_stl10.cpp nano)
build_test_app(test_task_mnist.cpp nano)
build_test_app(test_task_cifar10.cpp nano)
build_test_app(test_task_cifar100.cpp nano)
