#include "core/table.h"
#include "core/logger.h"
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

        if (table.rows() < 1 || table.cols() < 1)
        {
                log_error() << name << ": empty table!";
                return false;
        }

        return true;
}

static auto load_labels(const table_t& table, const size_t col)
{
        strings_t labels;
        for (size_t i = 0; i < table.rows(); ++ i)
        {
                const auto& row = table.row(i);
                labels.push_back(row.cell(col).m_data);
        }

        std::sort(labels.begin(), labels.end());
        labels.erase(
                std::unique(labels.begin(), labels.end()),
                labels.end());

        return labels;
}

mem_csv_task_t::mem_csv_task_t(string_t name, string_t path, const type task_type, indices_t target_columns) :
        mem_tensor_task_t(make_dims(1, 1, 1), make_dims(1, 1, 1), 10),
        m_name(std::move(name)),
        m_path(std::move(path)),
        m_type(task_type),
        m_target_columns(std::move(target_columns))
{
}

void mem_csv_task_t::from_json(const json_t& json)
{
        nano::from_json(json, "path", m_path, "folds", m_folds);
        reconfig(1, 1);
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

        // check target columns
        if (m_target_columns.empty())
        {
                log_error() << m_name << ": missing target columns!";
                return false;
        }
        if (m_target_columns.size() >= table.cols())
        {
                log_error() << m_name << ": too many target columns "
                        << m_target_columns.size() << "/" << table.cols() << "!";
                return false;
        }

        const auto it_col = std::find_if(m_target_columns.begin(), m_target_columns.end(),
                [&] (const auto& col) { return col >= table.cols(); });

        if (it_col != m_target_columns.end())
        {
                log_error() << m_name << ": invalid target column " << (*it_col) << "/" << table.cols() << "!";
                return false;
        }

        // load dataset
        switch (m_type)
        {
        case type::regression:
                return populate_regression(table);

        case type::classification:
                if (m_target_columns.size() != 1)
                {
                        log_error() << m_name << ": expecting a single target column for classification!";
                        return false;
                }
                return populate_classification(table);

        default:
                log_error() << m_name << ": invalid task type!";
                return false;
        }
}

bool mem_csv_task_t::populate_regression(const table_t& table)
{
        reconfig(
                static_cast<tensor_size_t>(table.cols() - m_target_columns.size()),
                static_cast<tensor_size_t>(m_target_columns.size()));

        // load samples
        strings_t labels(table.rows());
        tensor3ds_t inputs(table.rows(), tensor3d_t{idims()});
        tensor3ds_t targets(table.rows(), tensor3d_t{odims()});

        for (size_t r = 0, rows = table.rows(), cols = table.cols(); r < rows; ++ r)
        {
                auto& input = inputs[r];
                auto& target = targets[r];

                auto input_index = 0;
                auto target_index = 0;

                const auto& row = table.row(r);
                for (size_t c = 0; c < cols; ++ c)
                {
                        const auto& data = row.cell(c).m_data;
                        if (is_target(c))
                        {
                                target(target_index ++, 0, 0) = from_string<scalar_t>(data);
                        }
                        else
                        {
                                input(input_index ++, 0, 0) = from_string<scalar_t>(data);
                        }
                }
        }

        // scale inputs to improve numerical robustness
        scale(m_name, inputs);

        // OK
        return populate(inputs, targets, labels);
}

bool mem_csv_task_t::populate_classification(const table_t& table)
{
        const auto unique_labels = load_labels(table, m_target_columns[0]);
        log_info() << m_name << ": loaded " << join(unique_labels) << " labels.";

        reconfig(
                static_cast<tensor_size_t>(table.cols()) - 1,
                static_cast<tensor_size_t>(unique_labels.size()));

        // load samples
        strings_t labels(table.rows());
        tensor3ds_t inputs(table.rows(), tensor3d_t{idims()});
        tensor3ds_t targets(table.rows(), tensor3d_t{odims()});

        for (size_t r = 0, rows = table.rows(), cols = table.cols(); r < rows; ++ r)
        {
                auto& input = inputs[r];
                auto& target = targets[r];

                auto input_index = 0;

                const auto& row = table.row(r);
                for (size_t c = 0; c < cols; ++ c)
                {
                        const auto& data = row.cell(c).m_data;
                        if (is_target(c))
                        {
                                const auto itl = std::find(unique_labels.begin(), unique_labels.end(), data);
                                assert(itl != unique_labels.end());
                                labels[r] = data;
                                target.vector() = class_target(nano::size(odims()), itl - unique_labels.begin());
                        }
                        else
                        {
                                input(input_index ++, 0, 0) = from_string<scalar_t>(data);
                        }
                }
        }

        // scale inputs to improve numerical robustness
        scale(m_name, inputs);

        // OK
        return populate(inputs, targets, labels);
}

bool mem_csv_task_t::populate(const tensor3ds_t& inputs, const tensor3ds_t& targets, const strings_t& labels)
{
        assert(inputs.size() == targets.size());
        assert(inputs.size() == labels.size());

        for (size_t i = 0; i < inputs.size(); ++ i)
        {
                const auto hash = i;
                add_chunk(inputs[i], hash);
        }

        for (size_t f = 0; f < m_folds; ++ f)
        {
                const auto protocols = split3(inputs.size(),
                        protocol::train, m_train_percentage, protocol::valid, m_valid_percentage, protocol::test);

                for (size_t i = 0; i < inputs.size(); ++ i)
                {
                        const auto fold = fold_t{f, protocols[i]};
                        add_sample(fold, i, targets[i], labels[i]);
                }
        }

        // OK
        log_info() << m_name << ": loaded " << (size() / fsize()) << " samples with "
                << nano::size(idims()) << " attributes.";
        return true;
}
