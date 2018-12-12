#pragma once

#include "model.h"
#include "gboost.h"
#include "solver.h"
#include "core/tuner.h"
#include "core/tpool.h"
#include "core/logger.h"
#include "core/numeric.h"
#include "core/algorithm.h"
#include "wlearner_stump.h"
#include <iomanip>

namespace nano
{
        ///
        /// \brief Gradient Boosting with stumps as weak learners.
        /// see "The Elements of Statistical Learning", by Trevor Hastie, Robert Tibshirani, Jerome Friedman
        /// see "Greedy Function Approximation: A Gradient Boosting Machine", by Jerome Friedman
        /// see "Stochastic Gradient Boosting", by Jerome Friedman
        ///
        class gboost_stump_t final : public model_t
        {
        public:

                gboost_stump_t() = default;

                void to_json(json_t&) const override;
                void from_json(const json_t&) override;

                tensor3d_t output(const tensor3d_t& input) const override;
                training_t train(const task_t&, const size_t fold, const loss_t&) override;

                bool save(obstream_t&) const override;
                bool load(ibstream_t&) override;

                tensor3d_dim_t idims() const override { return m_idims; }
                tensor3d_dim_t odims() const override { return m_odims; }

        private:

                using stumps_t = std::vector<wlearner_stump_t>;

                template <typename tloss>
                training_t train(const task_t& task, const size_t fold, const loss_t& loss, const tuner_t& tuner)
                {
                        training_t result;
                        for (const auto& config : tuner.get(tuner.n_configs()))
                        {
                                const auto config_result = train_config<tloss>(task, fold, loss, config);
                                if (config_result.first < result)
                                {
                                        result = config_result.first;
                                        m_stumps = config_result.second;
                                }
                        }

                        log_info() << ">>>" << std::setprecision(4)
                                << " tr=" << result.optimum().m_train
                                << ",vd=" << result.optimum().m_valid
                                << ",te=" << result.optimum().m_test
                                << ",round=" << result.optimum().m_epoch
                                << "," << result.config() << ".";

                        return result;
                }

                template <typename tloss>
                std::pair<training_t, stumps_t> train_config(
                        const task_t& task, const size_t fold, const loss_t& loss, const json_t& json) const
                {
                        scalar_t lambda = 0, shrinkage = 0, subsampling = 0;
                        nano::from_json(json, "lambda", lambda, "shrinkage", shrinkage, "subsampling", subsampling);

                        // create loss functions
                        tloss loss_tr(task, fold_t{fold, protocol::train}, loss, lambda);
                        tloss loss_vd(task, fold_t{fold, protocol::valid}, loss, lambda);
                        tloss loss_te(task, fold_t{fold, protocol::test}, loss, lambda);

                        // check if the solver is properly set
                        rsolver_t solver;
                        critical(solver = get_solvers().get(m_solver),
                                strcat("search solver (", m_solver, ")"));

                        timer_t timer;

                        auto config = json.dump();
                        config = nano::replace(config, "\"", "");
                        config = nano::replace(config, ",}", "");
                        config = nano::replace(config, "}", "");
                        config = nano::replace(config, "{", "");
                        config = nano::replace(config, ":", "=");

                        training_t result(config);

                        const auto status = update_result(loss_tr, loss_vd, loss_te, timer, 0, result);

                        log_info() << std::setprecision(4) << "[" << 0 << "/" << m_rounds
                                << "]:tr=" << result.history().rbegin()->m_train
                                << ",vd=" << result.history().rbegin()->m_valid << "|" << status
                                << ",te=" << result.history().rbegin()->m_test
                                << "," << config << ".";

                        // add weak learners at each boosting round ...
                        stumps_t stumps;
                        for (auto round = 0; round < m_rounds && !result.is_done(); ++ round)
                        {
                                // fit stump
                                wlearner_stump_t stump;
                                stump.fit(task, fold_t{fold, protocol::train}, loss_tr.residuals(), m_wtype);
                                loss_tr.wlearner(stump);

                                // line-search
                                const auto epsilon = epsilon2<scalar_t>();
                                const auto x0 = vector_t{vector_t::Zero(loss_tr.size())};
                                const auto state = solver->minimize(100, epsilon, loss_tr, x0);
                                const auto step = state.x(0);

                                stump.scale(step);
                                stump.scale(shrinkage);
                                stumps.push_back(stump);

                                // update current predictions
                                loss_tr.add_wlearner(stump);
                                loss_vd.add_wlearner(stump);
                                loss_te.add_wlearner(stump);

                                const auto status = update_result(loss_tr, loss_vd, loss_te, timer, round + 1, result);

                                log_info()
                                        << std::setprecision(4)
                                        << "[" << (round + 1) << "/" << m_rounds
                                        << "]:tr=" << result.last().m_train
                                        << ",vd=" << result.last().m_valid << "|" << status
                                        << ",te=" << result.last().m_test
                                        << std::setprecision(2)
                                        << ",stump=(f=" << stump.feature() << ",t=" << stump.threshold() << ")"
                                        << std::setprecision(4)
                                        << ",solver=(" << state.m_status << ",i=" << state.m_iterations
                                        << ",x=" << state.x(0)
                                        << ",f=" << state.f << ",g=" << state.convergence_criteria() << ").";
                        }

                        // keep only the stumps up to optimum epoch (on the validation dataset)
                        stumps.erase(
                                stumps.begin() + result.optimum().m_epoch,
                                stumps.end());

                        return std::make_pair(result, stumps);
                }

                template <typename tloss>
                auto update_result(const tloss& loss_tr, const tloss& loss_vd, const tloss& loss_te,
                        const timer_t& timer, const int epoch, training_t& result) const
                {
                        const auto measure_tr = training_t::measurement_t{loss_tr.value(), loss_tr.error()};
                        const auto measure_vd = training_t::measurement_t{loss_vd.value(), loss_vd.error()};
                        const auto measure_te = training_t::measurement_t{loss_te.value(), loss_te.error()};

                        return result.update(
                                training_t::state_t{timer.milliseconds(), epoch, measure_tr, measure_vd, measure_te},
                                m_patience);
                }

        private:

                // attributes
                tensor3d_dim_t  m_idims{{0, 0, 0}};                     ///< input dimensions
                tensor3d_dim_t  m_odims{{0, 0, 0}};                     ///< output dimensions
                int             m_rounds{0};                            ///< number of boosting rounds
                int             m_patience{0};                          ///< number of epochs before overfitting
                string_t        m_solver{"cgd"};                        ///< solver to use for line-search
                cumloss         m_cumloss{cumloss::average};            ///<
                wlearner_type   m_wtype{wlearner_type::discrete};       ///<
                shrinkage       m_shrinkage{shrinkage::off};            ///<
                subsampling     m_subsampling{subsampling::off};        ///<
                stumps_t        m_stumps;                               ///< boosted weak learners
        };
}
