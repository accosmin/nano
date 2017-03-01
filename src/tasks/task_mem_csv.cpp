#include "class.h"
#include "logger.h"
#include "text/table.h"
#include "task_mem_csv.h"
#include "text/to_params.h"
#include "text/from_params.h"

namespace nano
{
        bool mem_csv_task_t::load_classification(const string_t& path, const string_t& task_name,
                const size_t expected_samples,
                const scalars_t& scales, const strings_t& labels, const size_t label_col)
        {
                const auto expected_features = static_cast<size_t>(nano::size(idims()));

                // load CSV
                const auto csv_delim = ",";
                const auto csv_header = false;

                table_t table;
                for (size_t col = 0, f = 0; col < expected_features + 1; ++ col)
                {
                        if (col == label_col)
                        {
                                table.header() << "class";
                        }
                        else
                        {
                                table.header() << ("f" + to_string(++ f));
                        }
                }

                log_info() << task_name << ": loading file <" << path << "> ...";
                if (!table.load(path, csv_delim, csv_header))
                {
                        log_error() << task_name << ": failed to load file <" << path << ">!";
                        return false;
                }
                if (table.rows() != expected_samples)
                {
                        log_error() << task_name << ": invalid number of samples!";
                        return false;
                }
                if (table.cols() != expected_features + 1)
                {
                        log_error() << task_name << ": invalid number of columns!";
                        return false;
                }
                if (scales.size() != expected_features)
                {
                        log_error() << task_name << ": invalid number of scaling factors";
                        return false;
                }

                // load samples
                for (size_t i = 0; i < table.rows(); ++ i)
                {
                        const auto& row = table.row(i);

                        const auto cc = row.value(label_col);
                        const auto itc = std::find(labels.begin(),labels.end(), cc);
                        if (itc == labels.end())
                        {
                                log_error() << task_name << ": invalid class <" << cc << ">!";
                                return false;
                        }

                        const auto make_sample = [this, row = std::ref(row), scales = std::ref(scales), label_col = label_col] ()
                        {
                                tensor3d_t sample(idims());
                                size_t col = 0;
                                for (auto k = 0; k < sample.size(); ++ k, ++ col)
                                {
                                        if (col == label_col)
                                        {
                                                ++ col;
                                        }
                                        const auto value = from_string<scalar_t>(row.get().value(col));
                                        sample(k, 0, 0) = value * scales.get()[static_cast<size_t>(k)];
                                }
                                return sample;
                        };

                        const auto hash = i;
                        const auto fold = make_fold(0);
                        const auto sample = make_sample();
                        const auto target = class_target(itc - labels.begin(), nano::size(odims()));

                        add_chunk(sample, hash);
                        add_sample(fold, i, target, cc);
                }

                // OK
                log_info() << task_name << ": loaded " << expected_samples << " samples.";
                return true;
        }
}
