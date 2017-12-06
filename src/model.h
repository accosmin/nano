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
                ///     (useful for quickly creating feed-forward networks)
                ///
                template <typename... tnames>
                bool connect(const string_t& name1, const string_t& name2, const tnames&... names)
                {
                        return connect(name1, name2) && connect(name2, names...);
                }

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
                /// \brief configuration done
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
                /// \brief compute the model's gradient wrt inputs given its output
                ///
                tensor4d_cmap_t ginput(const tensor4d_t& odata);

                ///
                /// \brief retrieve timing information for all components
                ///
                probes_t probes() const;

                ///
                /// \brief print a short description of the model
                ///
                void describe() const;

                ///
                /// \brief returns the input/output dimensions
                ///
                tensor3d_dim_t idims() const { return inode().m_node->idims(); }
                tensor3d_dim_t odims() const { return onode().m_node->odims(); }

                ///
                /// \brief returns number of parameters (to optimize)
                ///
                tensor_size_t psize() const { return m_pdata.size(); }

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
                const cnode_t& inode() const { assert(!m_nodes.empty()); return *m_nodes.begin(); }
                const cnode_t& onode() const { assert(!m_nodes.empty()); return *m_nodes.rbegin(); }

                // attributes
                cnodes_t        m_nodes;
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
