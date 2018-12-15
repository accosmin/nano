#pragma once

#include "model.h"
#include "gboost.h"
#include "wlearner_stump.h"
#include "wlearner_linear.h"

namespace nano
{
        ///
        /// \brief Gradient Boosting.
        /// see "The Elements of Statistical Learning", by Trevor Hastie, Robert Tibshirani, Jerome Friedman
        /// see "Greedy Function Approximation: A Gradient Boosting Machine", by Jerome Friedman
        /// see "Stochastic Gradient Boosting", by Jerome Friedman
        ///
        template <typename tweak_learner>
        class model_gboost_t final : public model_t
        {
        public:

                model_gboost_t() = default;

                void to_json(json_t&) const override;
                void from_json(const json_t&) override;

                tensor3d_t output(const tensor3d_t& input) const override;
                training_t train(const task_t&, const size_t fold, const loss_t&) override;

                bool save(obstream_t&) const override;
                bool load(ibstream_t&) override;

                tensor3d_dim_t idims() const override { return m_idims; }
                tensor3d_dim_t odims() const override { return m_odims; }

        private:

                using wlearners_t = std::vector<tweak_learner>;

                // attributes
                tensor3d_dim_t  m_idims{{0, 0, 0}};                     ///< input dimensions
                tensor3d_dim_t  m_odims{{0, 0, 0}};                     ///< output dimensions
                int             m_rounds{0};                            ///< number of boosting rounds
                int             m_patience{0};                          ///< number of epochs before overfitting
                string_t        m_solver{"cgd"};                        ///< solver to use for line-search
                cumloss         m_cumloss{cumloss::average};            ///<
                shrinkage       m_shrinkage{shrinkage::off};            ///<
                subsampling     m_subsampling{subsampling::off};        ///<
                wlearners_t     m_wlearners;                            ///< boosted weak learners
        };

        using model_gboost_linear_t = model_gboost_t<wlearner_linear_t>;
        using model_gboost_real_stump_t = model_gboost_t<wlearner_real_stump_t>;
        using model_gboost_discrete_stump_t = model_gboost_t<wlearner_discrete_stump_t>;
}
