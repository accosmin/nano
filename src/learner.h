#pragma once

#include "loss.h"
#include "task.h"
#include "core/stats.h"
#include "core/probe.h"
#include "core/ibstream.h"
#include "core/obstream.h"
#include "trainer_state.h"

namespace nano
{
        class learner_t;
        using learner_factory_t = factory_t<loss_t>;
        using rlearner_t = learner_factory_t::trobject;

        NANO_PUBLIC learner_factory_t& get_learners();

        ///
        /// \brief machine learning model.
        ///
        class NANO_PUBLIC learner_t : public json_configurable_t
        {
        public:

                ///
                /// \brief train the learner on the given task and using the given loss.
                ///
                virtual trainer_result_t train(const task_t&, const size_t fold, const loss_t&) const = 0;

                ///
                /// \brief compute the predictions for a set of inputs.
                ///
                virtual tensor4d_t output(const tensor4d_t& input) const = 0;
                stats_t error(const task_t&, const size_t fold, const loss_t&) const;
                stats_t value(const task_t&, const size_t fold, const loss_t&) const;

                ///
                /// \brief serialize a learner to disk
                ///
                static bool save(const string_t& path, const rlearner_t&);
                static rlearner_t load(const string_t& path);

                ///
                /// \brief serialize the learner to disk
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
        /// \brief check if the given learner is compatible with the given task.
        ///
        inline bool operator==(const learner_t& learner, const task_t& task)
        {
                return  learner.idims() == task.idims() &&
                        learner.odims() == task.odims();
        }

        inline bool operator!=(const learner_t& learner, const task_t& task)
        {
                return !(learner == task);
        }

        ///
        /// \brief convenience function to compute the size of the inputs / outputs of the given learner.
        ///
        inline auto isize(const learner_t& learner) { return nano::size(learner.idims()); }
        inline auto osize(const learner_t& learner) { return nano::size(learner.odims()); }

        inline auto isize(const learner_t& learner, const tensor_size_t count) { return count * isize(learner); }
        inline auto osize(const learner_t& learner, const tensor_size_t count) { return count * osize(learner); }

        inline auto idims(const learner_t& learner, const tensor_size_t count) { return cat_dims(count, learner.idims()); }
        inline auto odims(const learner_t& learner, const tensor_size_t count) { return cat_dims(count, learner.odims()); }
}
