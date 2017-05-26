#include <mutex>
#include "tasks/task_mnist.h"
#include "tasks/task_cifar10.h"
#include "tasks/task_cifar100.h"
#include "tasks/task_stl10.h"
#include "tasks/task_svhn.h"
#include "tasks/task_charset.h"
#include "tasks/task_iris.h"
#include "tasks/task_wine.h"

using namespace nano;

task_factory_t& nano::get_tasks()
{
        static task_factory_t manager;

        static std::once_flag flag;
        std::call_once(flag, [&m = manager] ()
        {
                m.add<mnist_task_t>("mnist", "MNIST (1x28x28 digit classification)");
                m.add<cifar10_task_t>("cifar10", "CIFAR-10 (3x32x32 object classification)");
                m.add<cifar100_task_t>("cifar100", "CIFAR-100 (3x32x32 object classification)");
                m.add<stl10_task_t>("stl10", "STL-10 (3x96x96 semi-supervised object classification)");
                m.add<svhn_task_t>("svhn", "SVHN (3x32x32 digit classification in the wild)");
                m.add<iris_task_t>("iris", "IRIS (iris flower classification)");
                m.add<wine_task_t>("wine", "WINE (wine classification)");
                m.add<charset_task_t>("synth-charset", "synthetic character classification");
        });

        return manager;
}
