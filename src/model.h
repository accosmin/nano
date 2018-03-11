#pragma once

#include "task.h"
#include "cnode.h"

namespace nano
{
        class model_t;
        using rmodel_t = std::unique_ptr<model_t>;

        ///
        /// \brief computation directed graph.
        ///
        class NANO_PUBLIC model_t
        {
        public:

                ///
                /// \brief constructors & asignment operators
                ///
                model_t() = default;
                model_t(const model_t&) = default;
                model_t(model_t&&) = default;
                model_t& operator=(model_t&&) = default;
                model_t& operator=(const model_t&) = delete;

                ///
                /// \brief copy the current object
                ///
                rmodel_t clone() const;

                ///
                /// \brief remove all computation nodes
                ///
                void clear();

                ///
                /// \brief add a new computation name given by its name, type and configuration
                ///
                bool add(const string_t& name, const string_t& type, json_reader_t&);
                bool add(const string_t& name, const string_t& type, const string_t& json);

                ///
                /// \brief connect two computation nodes such that the first one is an input of the second one
                ///
                bool connect(const string_t& name1, const string_t& name2);

                ///
                /// \brief connect a chain of computation nodes: name1->name2->names...
                ///     (e.g. useful for quickly creating feed-forward chains)
                ///
                template <typename... tnames>
                bool connect(const string_t& name1, const string_t& name2, const tnames&... names)
                {
                        return connect(name1, name2) && connect(name2, names...);
                }

                bool connect(const strings_t& names);

                ///
                /// \brief configure the computation graph using JSON
                ///
                bool config(json_reader_t&);
                bool config(const string_t& json);

                ///
                /// \brief serialize the computation graph to JSON
                ///
                void config(json_writer_t&) const;

                ///
                /// \brief mark the configuration done and verifies if the computation node form a valid graph:
                ///     - no cycles
                ///     - exactly one output (for now)
                ///
                /// NB: if all the conditions are met, the computation nodes are sorted topologically.
                ///
                bool done();

                ///
                /// \brief resize to process the given input/output size
                ///
                bool resize(const tensor3d_dim_t& idims, const tensor3d_dim_t& odims);

                ///
                /// \brief serialize model to disk
                ///
                bool save(const string_t& path) const;
                bool load(const string_t& path);

                ///
                /// \brief set parameters
                ///
                void params(const vector_t&);

                ///
                /// \brief set parameters to random values
                ///
                void random();

                ///
                /// \brief compute the model's output given its input
                ///
                tensor4d_cmap_t output(const tensor4d_t& idata);

                ///
                /// \brief compute the model's gradient wrt parameters given its output
                ///
                const vector_t& gparam(const tensor4d_t& odata);

                ///
                /// \brief retrieve timing information for all components
                ///
                probes_t probes() const;

                ///
                /// \brief print a short description of the model
                ///
                void describe() const;

                ///
                /// \brief returns the input/output/parameters dimensions
                ///
                tensor3d_dim_t idims() const { return m_idims; }
                tensor3d_dim_t odims() const { return m_odims; }

                tensor_size_t psize() const { return m_pdata.size(); }
                tensor_size_t isize() const { return nano::size(idims()); }
                tensor_size_t osize() const { return nano::size(odims()); }

                ///
                /// \brief returns the current parameters and their gradient
                ///
                const vector_t& params() const { return m_pdata; }

        private:

                bool config_nodes(json_reader_t&);
                bool config_model(json_reader_t&);

                void allocate(const tensor_size_t count);
                tensor_size_t xsize(const tensor_size_t count) const;

                const vector_t& cxdata() { return m_xdata; }
                const vector_t& cpdata() { return m_pdata; }
                const vector_t& cgdata() { return m_gdata; }

                strings_t node_names(const indices_t& indices) const;

                size_t find_node(const string_t& name) const;
                const cnode_t& onode() const { assert(!m_nodes.empty()); return *m_nodes.rbegin(); }

                // attributes
                tensor3d_dim_t  m_idims{{0, 0, 0}};     ///< input dimensions
                tensor3d_dim_t  m_odims{{0, 0, 0}};     ///< output dimensions
                cnodes_t        m_nodes;                ///< computation nodes
                vector_t        m_pdata;                ///< current parameters
                vector_t        m_gdata;                ///< current gradient wrt parameters
                vector_t        m_xdata;                ///< current input-output buffers
                probe_t         m_probe_output;
                probe_t         m_probe_ginput;
                probe_t         m_probe_gparam;
        };

        ///
        /// \brief check if the given model is compatible with the given task.
        ///
        inline bool operator==(const model_t& model, const task_t& task)
        {
                return  model.idims() == task.idims() &&
                        model.odims() == task.odims();
        }

        inline bool operator!=(const model_t& model, const task_t& task)
        {
                return !(model == task);
        }

        ///
        /// \brief resize the model to process samples from the given task
        ///
        inline bool resize(model_t& model, const task_t& task)
        {
                return model.resize(task.idims(), task.odims());
        }

        ///
        /// \brief convenience function to compute the size of the inputs / outputs of the given model.
        ///
        inline auto isize(const model_t& model) { return nano::size(model.idims()); }
        inline auto osize(const model_t& model) { return nano::size(model.odims()); }

        inline auto isize(const model_t& model, const tensor_size_t count) { return count * isize(model); }
        inline auto osize(const model_t& model, const tensor_size_t count) { return count * osize(model); }

        inline auto idims(const model_t& model, const tensor_size_t count) { return cat_dims(count, model.idims()); }
        inline auto odims(const model_t& model, const tensor_size_t count) { return cat_dims(count, model.odims()); }
}
