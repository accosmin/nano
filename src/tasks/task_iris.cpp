#include "task_iris.h"

using namespace nano;

iris_task_t::iris_task_t() :
        mem_csv_task_t(tensor3d_dims_t{4, 1, 1}, tensor3d_dims_t{3, 1, 1}, 1),
        m_dir(string_t(std::getenv("HOME")) + "/experiments/databases/iris")
{
}

json_reader_t& iris_task_t::config(json_reader_t& reader)
{
        return reader.object("dir", m_dir);
}

json_writer_t& iris_task_t::config(json_writer_t& writer) const
{
        return writer.object("dir", m_dir);
}

bool iris_task_t::populate()
{
        const auto path = m_dir + "/iris.data";
        const auto task_name = "IRIS";

        const auto expected_samples = size_t(150);
        const auto labels = strings_t{"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
        const auto label_column = size_t(4);

        return mem_csv_task_t::load_classification(path, task_name, expected_samples, labels, label_column);
}
