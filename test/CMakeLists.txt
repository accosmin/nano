# setup CMake to run tests
enable_testing()

# build unit tests
make_test(test_average.cpp "")
make_test(test_cubic.cpp "")
make_test(test_gauss.cpp "")
make_test(test_momentum.cpp "")
make_test(test_quadratic.cpp "")
make_test(test_random.cpp "")
make_test(test_stats.cpp "")
make_test(test_tensor.cpp "")
make_test(test_tune.cpp "")
make_test(test_chrono.cpp "")

make_test(test_stoch.cpp nano)
make_test(test_batch.cpp nano)
make_test(test_functions.cpp nano)

make_test(test_text.cpp nano)
make_test(test_cmdline.cpp nano)
make_test(test_filesystem.cpp nano)

make_test(test_io.cpp nano)

make_test(test_thread_loop.cpp nano)
make_test(test_thread_pool.cpp nano)

make_test(test_geom.cpp "")
make_test(test_image.cpp nano)
make_test(test_vision.cpp "")

make_test(test_accumulator.cpp nano)
make_test(test_layers.cpp nano)
make_test(test_loss.cpp nano)
make_test(test_factory.cpp nano)
make_test(test_model.cpp nano)
make_test(test_trainer.cpp nano)
make_test(test_affine.cpp nano)
make_test(test_conv.cpp nano)
make_test(test_conv3d.cpp nano)
make_test(test_normalize.cpp nano)

make_test(test_charset.cpp nano)
make_test(test_iterator.cpp nano)

make_test(test_task_svhn.cpp nano)
make_test(test_task_iris.cpp nano)
make_test(test_task_wine.cpp nano)
make_test(test_task_stl10.cpp nano)
make_test(test_task_mnist.cpp nano)
make_test(test_task_cifar10.cpp nano)
make_test(test_task_cifar100.cpp nano)
