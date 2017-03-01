#include "class.h"
#include "logger.h"
#include "task_iris.h"
#include "text/table.h"
#include "text/to_params.h"
#include "text/from_params.h"

namespace nano
{
        static const strings_t tlabels =
        {
                "Iris-setosa",
                "Iris-versicolor",
                "Iris-virginica"
        };

        iris_task_t::iris_task_t(const string_t& config) :
                mem_tensor_task_t(dim3d_t{4, 1, 1}, dim3d_t{3, 1, 1}, 1,
                to_params(config, "dir", string_t(std::getenv("HOME")) + "/experiments/databases/iris"))
        {
        }

        bool iris_task_t::populate()
        {
                const string_t dir = from_params<string_t>(config(), "dir");
                const string_t file = dir + "/iris.data";
                const size_t n_samples = 150;

                // load CSV
                const auto csv_delim = ",";
                const auto csv_header = false;

                table_t table;
                table.header() << "f0" << "f1" << "f2" << "f3" << "class";

                log_info() << "IRIS: loading file <" << file << "> ...";
                if (!table.load(file, csv_delim, csv_header))
                {
                        log_error() << "IRIS: failed to load file <" << file << ">!";
                        return false;
                }
                if (table.rows() != n_samples)
                {
                        log_error() << "IRIS: invalid number of samples!";
                        return false;
                }
                if (table.cols() != static_cast<size_t>(nano::size(idims()) + 1))
                {
                        log_error() << "IRIS: invalid number of columns!";
                        return false;
                }

                // load samples
                for (size_t i = 0; i < table.rows(); ++ i)
                {
                        const auto& row = table.row(i);

                        const auto cc = row.value(table.cols() - 1);
                        const auto itc = std::find(tlabels.begin(), tlabels.end(), cc);
                        if (itc == tlabels.end())
                        {
                                log_error() << "IRIS: invalid class <" << cc << ">!";
                                return false;
                        }

                        const auto make_sample = [this, row = std::ref(row)] (const size_t offset)
                        {
                                tensor3d_t sample(idims());
                                for (auto k = 0; k < sample.size(); ++ k)
                                {
                                        sample(k, 0, 0) = from_string<scalar_t>(row.get().value(offset + static_cast<size_t>(k)));
                                }
                                return sample;
                        };

                        const auto hash = i;
                        const auto fold = make_fold(0);
                        const auto sample = make_sample(0);
                        const auto target = class_target(itc - tlabels.begin(), nano::size(odims()));

                        add_chunk(sample, hash);
                        add_sample(fold, i, target, cc);
                }

                // OK
                log_info() << "IRIS: loaded " << n_samples << " samples.";
                return true;
        }
}
