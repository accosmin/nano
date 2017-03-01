#include "class.h"
#include "logger.h"
#include "task_wine.h"
#include "text/table.h"
#include "text/to_params.h"
#include "text/from_params.h"

namespace nano
{
        static const strings_t tlabels =
        {
                "1",
                "2",
                "3"
        };

        wine_task_t::wine_task_t(const string_t& config) :
                mem_tensor_task_t(dim3d_t{13, 1, 1}, dim3d_t{3, 1, 1}, 1,
                to_params(config, "dir", string_t(std::getenv("HOME")) + "/experiments/databases/wine"))
        {
        }

        bool wine_task_t::populate()
        {
                const string_t dir = from_params<string_t>(config(), "dir");
                const string_t file = dir + "/wine.data";
                const size_t n_samples = 178;

                // load CSV
                const auto csv_delim = ",";
                const auto csv_header = false;

                table_t table;
                table.header() << "class" << "f0" << "f1" << "f2" << "f3" << "f4" << "f5" << "f6" << "f7" << "f8" << "f9" << "f10" << "f11" << "f12";

                log_info() << "WINE: loading file <" << file << "> ...";
                if (!table.load(file, csv_delim, csv_header))
                {
                        log_error() << "WINE: failed to load file <" << file << ">!";
                        return false;
                }
                if (table.rows() != n_samples)
                {
                        log_error() << "WINE: invalid number of samples!";
                        return false;
                }
                if (table.cols() != static_cast<size_t>(nano::size(idims()) + 1))
                {
                        log_error() << "WINE: invalid number of columns!";
                        return false;
                }

                // load samples
                for (size_t i = 0; i < table.rows(); ++ i)
                {
                        const auto& row = table.row(i);

                        const auto cc = row.value(0);
                        const auto itc = std::find(tlabels.begin(), tlabels.end(), cc);
                        if (itc == tlabels.end())
                        {
                                log_error() << "WINE: invalid class <" << cc << ">!";
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
                        const auto sample = make_sample(1);
                        const auto target = class_target(itc - tlabels.begin(), nano::size(odims()));

                        add_chunk(sample, hash);
                        add_sample(fold, i, target, cc);
                }

                // OK
                log_info() << "WINE: loaded " << n_samples << " samples.";
                return true;
        }
}
