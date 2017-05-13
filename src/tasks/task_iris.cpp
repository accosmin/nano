#include "task_iris.h"
#include "text/to_params.h"
#include "text/from_params.h"

namespace nano
{
        iris_task_t::iris_task_t(const string_t& config) :
                mem_csv_task_t(tensor3d_dims_t{4, 1, 1}, tensor3d_dims_t{3, 1, 1}, 1,
                to_params(config, "dir", string_t(std::getenv("HOME")) + "/experiments/databases/iris"))
        {
        }

        bool iris_task_t::populate()
        {
                const auto path = from_params<string_t>(config(), "dir") + "/iris.data";
                const auto task_name = "IRIS";

                const auto expected_samples = size_t(150);
                const auto labels = strings_t{"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
                const auto label_column = size_t(4);

                return mem_csv_task_t::load_classification(path, task_name, expected_samples, labels, label_column);
        }
}
