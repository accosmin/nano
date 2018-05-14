#include "logger.h"
#include "text/table.h"
#include "task_mem_csv.h"

using namespace nano;

static void scale(const string_t& name, tensor3ds_t& samples)
{
        const auto dims = samples.begin()->size();

        vector_t maximums = vector_t::Constant(dims, 1);
        for (const auto& sample : samples)
        {
                assert(maximums.size() == sample.array().size());
                maximums.array() = maximums.array().max(sample.array().abs());
        }

        const vector_t scale = 1 / maximums.array();
        for (auto& sample : samples)
        {
                sample.array() *= scale.array();
        }

        log_info() << name << ": scaled using [" << maximums.transpose() << "].";
}

static bool load_csv(const string_t& path, const string_t& name, table_t& table)
{
        const auto csv_delim = ",";
        const auto csv_header = false;

        log_info() << name << ": loading file <" << path << "> ...";
        if (!table.load(path, csv_delim, csv_header))
        {
                log_error() << name << ": failed to load file <" << path << ">!";
                return false;
        }

        return true;
}

static bool load_labels(const string_t& name, const table_t& table, const size_t label_column, strings_t& labels)
{
        for (size_t i = 0; i < table.rows(); ++ i)
        {
                const auto& row = table.row(i);
                if (label_column >= row.cols())
                {
                        log_error() << name << ": invalid label column " << label_column << "/" << row.cols() << "!";
                        return false;
                }

                labels.push_back(row.cell(label_column).m_data);
        }

        std::sort(labels.begin(), labels.end());
        labels.erase(
                std::unique(labels.begin(), labels.end()),
                labels.end());

        return true;
}

mem_csv_task_t::mem_csv_task_t(const string_t& name, const string_t& path, const size_t label_column) :
        mem_tensor_task_t(make_dims(1, 1, 1), make_dims(1, 1, 1), 1),
        m_name(name),
        m_path(path),
        m_label_column(label_column)
{
}

void mem_csv_task_t::from_json(const json_t& json)
{
        nano::from_json(json, "path", m_path, "folds", m_folds);
}

void mem_csv_task_t::to_json(json_t& json) const
{
        nano::to_json(json, "path", m_path, "folds", m_folds);
}

bool mem_csv_task_t::populate()
{
        // load CSV
        table_t table;
        if (!load_csv(m_path, m_name, table))
        {
                return false;
        }

        strings_t labels;
        if (!load_labels(m_name, table, m_label_column, labels))
        {
                return false;
        }

        const auto n_samples = static_cast<tensor_size_t>(table.rows());
        const auto n_labels = static_cast<tensor_size_t>(labels.size());
        const auto n_attributes = static_cast<tensor_size_t>(table.cols()) - 1;

        reconfig(
                make_dims(n_attributes, 1, 1),
                make_dims(n_labels, 1, 1),
                m_folds);

        // load samples
        tensor3ds_t samples(table.rows(), tensor3d_t{idims()});
        std::vector<tensor_size_t> class_indices(table.rows());
        for (size_t i = 0; i < table.rows(); ++ i)
        {
                auto& sample = samples[i];
                auto& class_index = class_indices[i];

                const auto& row = table.row(i);
                for (size_t c = 0; c < table.cols(); ++ c)
                {
                        const auto& data = row.cell(c).m_data;
                        if (c == m_label_column)
                        {
                                const auto itl = std::find(labels.begin(), labels.end(), data);
                                assert(itl != labels.end());
                                class_index = itl - labels.begin();
                        }
                        else
                        {
                                sample(static_cast<tensor_size_t>(c ? c - 1 : c), 0, 0) = from_string<scalar_t>(data);
                        }
                }
        }

        // scale inputs to improve numerical robustness
        if (!samples.empty() && samples.begin()->size() > 0)
        {
                scale(m_name, samples);
        }

        // setup task
        for (size_t i = 0; i < samples.size(); ++ i)
        {
                const auto hash = i;
                add_chunk(samples[i], hash);
        }

        for (size_t f = 0; f < m_folds; ++ f)
        {
                const auto protocols = split3(samples.size(),
                        protocol::train, m_train_percentage, protocol::valid, m_valid_percentage, protocol::test);

                for (size_t i = 0; i < samples.size(); ++ i)
                {
                        const auto fold = fold_t{f, protocols[i]};
                        const auto target = class_target(class_indices[i], nano::size(odims()));
                        add_sample(fold, i, target, labels[static_cast<size_t>(class_indices[i])]);
                }
        }

        // OK
        log_info() << m_name << ": loaded " << n_samples << " samples with "
                << n_attributes << " attributes and " << join(labels) << " labels.";
        return true;
}
