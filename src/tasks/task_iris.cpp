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
                if (table.cols() != 5)
                {
                        log_error() << "IRIS: invalid number of columns!";
                        return false;
                }

                // load samples
                for (size_t i = 0; i < table.rows(); ++ i)
                {
                        const auto& row = table.row(i);

                        const auto f0 = from_string<scalar_t>(row.value(0));
                        const auto f1 = from_string<scalar_t>(row.value(1));
                        const auto f2 = from_string<scalar_t>(row.value(2));
                        const auto f3 = from_string<scalar_t>(row.value(3));
                        const auto cc = row.value(4);

                        const auto itc = std::find(tlabels.begin(), tlabels.end(), cc);
                        if (itc == tlabels.end())
                        {
                                log_error() << "IRIS: invalid class <" << cc << ">!";
                                return false;
                        }

                        const auto make_sample = [=] ()
                        {
                                tensor3d_t sample(4, 1, 1);
                                sample(0, 0, 0) = f0;
                                sample(1, 0, 0) = f1;
                                sample(2, 0, 0) = f2;
                                sample(3, 0, 0) = f3;
                                return sample;
                        };

                        const auto hash = i;
                        const auto fold = make_fold(0);
                        const auto sample = make_sample();
                        const auto target = class_target(itc - tlabels.begin(), nano::size(odims()));

                        add_chunk(sample, hash);
                        add_sample(fold, i, target, cc);
                }

                // OK
                log_info() << "IRIS: loaded " << n_samples << " samples.";
                return true;
        }
}
