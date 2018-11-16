#pragma once

#include "loss.h"
#include "task.h"
#include "core/probe.h"
#include "trainer_state.h"

namespace nano
{
        class model_t;
        using model_factory_t = factory_t<model_t>;
        using rmodel_t = model_factory_t::trobject;

        NANO_PUBLIC model_factory_t& get_models();

        class ibstream_t;
        class obstream_t;

        ///
        /// \brief machine learning model.
        ///
        class NANO_PUBLIC model_t : public json_configurable_t
        {
        public:

                ///
                /// \brief train the model on the given task and using the given loss.
                ///
                virtual trainer_result_t train(const task_t&, const size_t fold, const loss_t&) = 0;

                ///
                /// \brief compute the prediction for an input.
                ///
                virtual tensor3d_t output(const tensor3d_t& input) const = 0;

                ///
                /// \brief compute statistics for a given task.
                ///
                struct evaluate_t
                {
                        operator bool() const { return m_values && m_errors; }

                        // attributes
                        stats_t         m_values;       ///< loss values
                        stats_t         m_errors;       ///< error values
                        milliseconds_t  m_millis{0};    ///< average milliseconds per sample
                };

                evaluate_t evaluate(const task_t&, const fold_t&, const loss_t&) const;

                ///
                /// \brief serialize a model to disk
                ///
                static bool save(const string_t& path, const string_t& id, const model_t&);
                static rmodel_t load(const string_t& path);

                ///
                /// \brief serialize the model to disk
                ///
                virtual bool save(obstream_t&) const = 0;
                virtual bool load(ibstream_t&) = 0;

                ///
                /// \brief returns the expected input/output dimensions
                ///
                virtual tensor3d_dim_t idims() const = 0;
                virtual tensor3d_dim_t odims() const = 0;

                ///
                /// \brief retrieve timing information for all components
                ///
                virtual probes_t probes() const = 0;
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
}
