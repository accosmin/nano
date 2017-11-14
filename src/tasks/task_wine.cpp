#include "task_wine.h"

using namespace nano;

wine_task_t::wine_task_t() :
        mem_csv_task_t(tensor3d_dims_t{13, 1, 1}, tensor3d_dims_t{3, 1, 1}, 1),
        m_dir(string_t(std::getenv("HOME")) + "/experiments/databases/wine")
{
}

json_reader_t& wine_task_t::config(json_reader_t& reader)
{
        return reader.object("dir", m_dir);
}

json_writer_t& wine_task_t::config(json_writer_t& writer) const
{
        return writer.object("dir", m_dir);
}

bool wine_task_t::populate()
{
        const auto path = m_dir + "/wine.data";
        const auto task_name = "WINE";

        const auto expected_samples = size_t(178);
        const auto labels = strings_t{"1", "2", "3"};
        const auto label_column = size_t(0);

        return mem_csv_task_t::load_classification(path, task_name, expected_samples, labels, label_column);
}
