#pragma once

#include "model.h"
#include "stump.h"
#include "gboost.h"

namespace nano
{
        ///
        /// \brief Gradient Boosting with stumps as weak learners.
        ///     see Friedman, J. H. (February 1999). "Greedy Function Approximation: A Gradient Boosting Machine" (PDF).
        ///     see Friedman, J. H. (March 1999). "Stochastic Gradient Boosting" (PDF).
        ///
        class gboost_stump_t final : public model_t
        {
        public:

                gboost_stump_t() = default;

                void to_json(json_t&) const override;
                void from_json(const json_t&) override;

                trainer_result_t train(const task_t&, const size_t fold, const loss_t&) override;

                tensor3d_t output(const tensor3d_t& input) const override;

                bool save(obstream_t&) const override;
                bool load(ibstream_t&) override;

                tensor3d_dim_t idims() const override { return m_idims; }
                tensor3d_dim_t odims() const override { return m_odims; }

                probes_t probes() const override;

        private:

                tensor4d_t residuals(
                        const task_t&, const fold_t&, const loss_t&, const tensor4d_t& outputs, const scalar_t lambda) const;

                std::pair<trainer_result_t, stumps_t> train(
                        const task_t&, const size_t fold, const loss_t&, const scalar_t lambda) const;

        private:

                // attributes
                tensor3d_dim_t  m_idims{{0, 0, 0}};                     ///< input dimensions
                tensor3d_dim_t  m_odims{{0, 0, 0}};                     ///< output dimensions
                int             m_rounds{0};                            ///< number of boosting rounds
                int             m_patience{0};                          ///< number of epochs before overfitting
                string_t        m_solver{"cgd"};                        ///< solver to use for line-search
                stump_type      m_stump_type{stump_type::discrete};     ///< stump type
                gboost_tune     m_gboost_tune{gboost_tune::none};       ///< regularization method
                stumps_t        m_stumps;                               ///< trained stumps
        };
}
