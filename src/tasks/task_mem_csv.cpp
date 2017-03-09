#include "class.h"
#include "logger.h"
#include "text/table.h"
#include "task_mem_csv.h"
#include "text/to_params.h"
#include "text/from_params.h"

namespace nano
{
        static void scale(const string_t& task_name, tensor3ds_t& samples)
        {
                const auto dims = samples.begin()->size();

                vector_t maximums(dims);
                for (const auto& sample : samples)
                {
                        maximums.array() = maximums.array().max(sample.array().abs());
                }

                const vector_t scale = 1 / maximums.array();
                for (auto& sample : samples)
                {
                        sample.array() *= scale.array();
                }

                log_info() << task_name << ": scaling using [" << maximums.transpose() << "]";
        }

        static bool load_csv(const string_t& path, const string_t& task_name,
                const size_t expected_samples, const size_t expected_features, table_t& table)
        {
                const auto csv_delim = ",";
                const auto csv_header = false;

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

                return true;
        }

        bool mem_csv_task_t::load_classification(const string_t& path, const string_t& task_name,
                const size_t expected_samples,
                const strings_t& labels, const size_t label_col)
        {
                const auto expected_features = static_cast<size_t>(nano::size(idims()));

                // load CSV
                table_t table;
                for (size_t col = 0, f = 0; col < expected_features + 1; ++ col)
                {
                        table.header() << (col == label_col ? "class" : ("f" + to_string(++ f)));
                }

                if (!load_csv(path, task_name, expected_samples, expected_features, table))
                {
                        return false;
                }

                // load samples
                tensor3ds_t samples;
                std::vector<tensor_index_t> class_indices;
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

                        const auto make_sample = [this, row = std::ref(row), label_col = label_col] ()
                        {
                                tensor3d_t sample(idims());
                                size_t col = 0;
                                for (auto k = 0; k < sample.size(); ++ k, ++ col)
                                {
                                        if (col == label_col)
                                        {
                                                ++ col;
                                        }
                                        sample(k, 0, 0) = from_string<scalar_t>(row.get().value(col));
                                }
                                return sample;
                        };

                        samples.push_back(make_sample());
                        class_indices.push_back(itc - labels.begin());
                }

                // scale inputs to improve numerical robustness
                scale(task_name, samples);

                // setup task
                for (size_t i = 0; i < samples.size(); ++ i)
                {
                        const auto hash = i;
                        const auto fold = make_fold(0);
                        const auto target = class_target(class_indices[i], nano::size(odims()));

                        add_chunk(samples[i], hash);
                        add_sample(fold, i, target, labels[static_cast<size_t>(class_indices[i])]);
                }

                // OK
                log_info() << task_name << ": loaded " << expected_samples << " samples.";
                return true;
        }
}
