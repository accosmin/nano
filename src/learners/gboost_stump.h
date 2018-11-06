#pragma once

#include "learner.h"

namespace nano
{
        ///
        /// \brief stump is a weak learner that compares the value of a selected feature with a threshold:
        ///     stump(x) = outputs(0) if x(feature) < threshold else v1(0)
        ///
        struct stump_t
        {
                // attributes
                tensor_size_t   m_feature{0};   ///< index of the selected feature
                scalar_t        m_threshold{0}; ///< threshold
                tensor4d_t      m_outputs;      ///< (2, #outputs) - predictions below and above the threshold
        };

        enum class stump_type
        {
                real,                           ///< stump \in R (no restriction)
                discrete,                       ///< stump \in {-1, +1}
        };

        // todo: generalize stump_t to use other features (e.g. Haar, HoG)
        // todo: implement different shrinkage methods: geometric or arithmetic decay (a single factor to tune)

        ///
        /// \brief Gradient Boosting with stumps as weak learners.
        ///     todo: add citations
        ///
        class gboost_stump_t final : public learner_t
        {
        public:

                enum class regularization
                {
                        none,                   ///<
                        adaptive,               ///< shrinkage per round using the validation dataset
                        shrinkage,              ///< global shrinkage (needs tuning)
                        vadaboost,              ///< VadaBoost (needs tuning)
                        // todo: Stochastic gradient boosting
                };

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

                using stumps_t = std::vector<stump_t>;

                // attributes
                tensor3d_dim_t  m_idims{{0, 0, 0}};                     ///< input dimensions
                tensor3d_dim_t  m_odims{{0, 0, 0}};                     ///< output dimensions
                int             m_rounds{0};                            ///< training: number of boosting rounds
                int             m_patience{0};                          ///< training: number of epochs before overfitting
                string_t        m_solver{"cgd"};                        ///< training: solver to use for line-search
                stump_type      m_stump_type{stump_type::discrete};     ///< training: stump type
                regularization  m_rtype{regularization::adaptive};      ///< training:
                stumps_t        m_stumps;                               ///< trained stumps
        };

        template <>
        inline enum_map_t<stump_type> enum_string<stump_type>()
        {
                return
                {
                        { stump_type::real,                     "real" },
                        { stump_type::discrete,                 "discrete" }
                };
        }

        template <>
        inline enum_map_t<gboost_stump_t::regularization> enum_string<gboost_stump_t::regularization>()
        {
                return
                {
                        { gboost_stump_t::regularization::none,         "none" },
                        { gboost_stump_t::regularization::adaptive,     "adaptive" },
                        { gboost_stump_t::regularization::shrinkage,    "shrinkage" },
                        { gboost_stump_t::regularization::vadaboost,    "vadaboost" }
                };
        }
}
