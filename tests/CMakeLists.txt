add_definitions(-DUTEST_WITH_EIGEN)

make_test(test_core_cast.cpp "")
make_test(test_core_stats.cpp "")
make_test(test_core_algorithm "")
make_test(test_core_chrono.cpp "")
make_test(test_core_geom2d.cpp "")
make_test(test_core_random.cpp "")
make_test(test_core_digraph.cpp "")
make_test(test_core_factory.cpp "")

make_test(test_core_io.cpp nano)
make_test(test_core_image.cpp nano)
make_test(test_core_tpool.cpp nano)
make_test(test_core_table.cpp nano)
make_test(test_core_tuner.cpp nano)
make_test(test_core_cmdline.cpp nano)

make_test(test_tensor_index.cpp "")
make_test(test_tensor_tensor.cpp "")
make_test(test_tensor_integral.cpp "")

make_test(test_wlearner_stump.cpp nano)
make_test(test_wlearner_table.cpp nano)
make_test(test_wlearner_linear.cpp nano)

make_test(test_gboost_loss.cpp nano)
make_test(test_model_gboost.cpp nano)

make_test(test_solvers.cpp nano)
make_test(test_functions.cpp nano)
make_test(test_solver_state.cpp nano)

make_test(test_loss.cpp nano)

make_test(test_task_svhn.cpp nano)
make_test(test_task_iris.cpp nano)
make_test(test_task_wine.cpp nano)
make_test(test_task_mnist.cpp nano)
make_test(test_task_affine.cpp nano)
make_test(test_task_cifar10.cpp nano)
make_test(test_task_cifar100.cpp nano)
make_test(test_task_cal_housing.cpp nano)
make_test(test_task_fashion_mnist.cpp nano)
