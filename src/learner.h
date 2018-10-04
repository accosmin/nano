#pragma once

#include "loss.h"
#include "task.h"
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
        class NANO_PUBLIC learner_t
        {
        public:

                ///
                /// \brief train the learner on the given task and using the given loss.
                ///     the training configuration is provided as json.
                ///
                virtual trainer_result_t train(const task_t&, const size_t fold, const loss_t&, const json_t&) const = 0;

                ///
                /// \brief
                ///
                virtual tensor4d_t predict(const tensor4d_t& input) const = 0;
                virtual stats_t<scalar_t> error(const task_t&, const size_t fold, const loss_t&) const;
                virtual stats_t<scalar_t> value(const task_t&, const size_t fold, const loss_t&) const;

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

                tensor_size_t isize() const { return nano::size(idims()); }
                tensor_size_t osize() const { return nano::size(odims()); }

                ///
                /// \brief retrieve timing information for all components
                ///
                virtual probes_t probes() const = 0;

        private:

                // attributes
                tensor3d_dim_t  m_idims{{0, 0, 0}};     ///< input dimensions
                tensor3d_dim_t  m_odims{{0, 0, 0}};     ///< output dimensions
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
