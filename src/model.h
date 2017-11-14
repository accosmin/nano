#pragma once

#include "task.h"
#include "layer.h"
#include "digraph.h"

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
                /// \brief
                ///
                model_t(const model_t&);
                model_t(model_t&&) = default;
                model_t& operator=(model_t&&) = default;
                model_t& operator=(const model_t&) = delete;

                ///
                /// \brief copy the current object
                ///
                rmodel_t clone() const;

                ///
                /// \brief resize to process new inputs
                ///
                bool config(const tensor3d_dims_t& idims, const tensor3d_dims_t& odims);

                ///
                /// \brief serialize model to disk
                ///
                bool save(const string_t& path) const;
                bool load(const string_t& path);

                ///
                /// \brief serialize parameters to memory
                ///
                vector_t params() const;
                void params(const vector_t&);

                ///
                /// \brief set parameters to random values
                ///
                void random();

                ///
                /// \brief compute the model's output given its input
                ///
                const tensor4d_t& output(const tensor4d_t& idata);

                ///
                /// \brief compute the model's gradient wrt parameters given its output
                ///
                const tensor1d_t& gparam(const tensor4d_t& odata);

                ///
                /// \brief compute the model's gradient wrt inputs given its output
                ///
                const tensor4d_t& ginput(const tensor4d_t& odata);

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
                tensor3d_dims_t idims() const;
                tensor3d_dims_t odims() const;

                ///
                /// \brief number of parameters (to optimize)
                ///
                tensor_size_t psize() const;

        private:

                using rlayers_t = std::vector<rlayer_t>;
                using digraph_t = nano::digraph_t<uint16_t>;

                // attributes
                rlayers_t       m_nodes;        ///< computation nodes
                digraph_t       m_graph;        ///< computation graph (aka dependency between nodes)
                tensor1d_t      m_gdata;        ///< gradient wrt parameters
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
        /// \brief convenience function to compute the size of the inputs / outputs of the given model.
        ///
        inline auto isize(const model_t& model) { return nano::size(model.idims()); }
        inline auto osize(const model_t& model) { return nano::size(model.odims()); }

        inline auto isize(const model_t& model, const tensor_size_t count) { return count * isize(model); }
        inline auto osize(const model_t& model, const tensor_size_t count) { return count * osize(model); }

        inline auto idims(const model_t& model, const tensor_size_t count) { return cat_dims(count, model.idims()); }
        inline auto odims(const model_t& model, const tensor_size_t count) { return cat_dims(count, model.odims()); }

        ///
        /// \brief convenience function to configure a model to process samples from the given task.
        ///
        inline bool config(model_t& model, const task_t& task)
        {
                return model.config(task.idims(), task.odims());
        }
}
