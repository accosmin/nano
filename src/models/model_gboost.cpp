#include <iomanip>
#include "solver.h"
#include "version.h"
#include "core/tuner.h"
#include "core/logger.h"
#include "core/random.h"
#include "model_gboost.h"
#include "core/ibstream.h"
#include "core/obstream.h"
#include "core/algorithm.h"
#include "gboost_loss_avg.h"
#include "gboost_loss_var.h"

using namespace nano;

template <typename tloss>
static auto update_result(const tloss& loss_tr, const tloss& loss_vd, const tloss& loss_te,
        const nano::timer_t& timer, const int epoch, const int patience, training_t& result)
{
        const auto measure_tr = training_t::measurement_t{loss_tr.value(), loss_tr.error()};
        const auto measure_vd = training_t::measurement_t{loss_vd.value(), loss_vd.error()};
        const auto measure_te = training_t::measurement_t{loss_te.value(), loss_te.error()};

        return result.update(
                training_t::state_t{timer.milliseconds(), epoch, measure_tr, measure_vd, measure_te},
                patience);
}

template <typename tloss, typename tweak_learner = typename tloss::weak_learner_t>
static auto train_config(
        const task_t& task, const size_t fold, const loss_t& loss, const json_t& json,
        const string_t& solver_id, const int rounds, const int patience)
{
        scalar_t lambda = 0, shrinkage = 0, subsampling = 0;
        nano::from_json(json, "lambda", lambda, "shrinkage", shrinkage, "subsampling", subsampling);

        // create loss functions
        tloss loss_tr(task, fold_t{fold, protocol::train}, loss, lambda);
        tloss loss_vd(task, fold_t{fold, protocol::valid}, loss, lambda);
        tloss loss_te(task, fold_t{fold, protocol::test}, loss, lambda);

        // check if the solver is properly set
        rsolver_t solver;
        critical(solver = get_solvers().get(solver_id),
                strcat("search solver (", solver_id, ")"));

        nano::timer_t timer;

        auto config = json.dump();
        config = nano::replace(config, "\"", "");
        config = nano::replace(config, ",}", "");
        config = nano::replace(config, "}", "");
        config = nano::replace(config, "{", "");
        config = nano::replace(config, ":", "=");

        training_t result(config);

        auto status = update_result(loss_tr, loss_vd, loss_te, timer, 0, patience, result);

        log_info() << std::setprecision(4) << "[" << 0 << "/" << rounds
                << "]:tr=" << result.history().rbegin()->m_train
                << ",vd=" << result.history().rbegin()->m_valid << "|" << status
                << ",te=" << result.history().rbegin()->m_test
                << "," << config << ".";

        // add weak learners at each boosting round ...
        std::vector<tweak_learner> wlearners;
        for (auto round = 0; round < rounds && !result.is_done(); ++ round)
        {
                // subsampling
                const auto indices = nano::sample_without_replacement(
                        task.size(fold_t{fold, protocol::train}),
                        static_cast<size_t>(subsampling));

                // fit weak learner
                tweak_learner wlearner;
                wlearner.fit(task, fold_t{fold, protocol::train}, loss_tr.gradients(), indices);
                loss_tr.wlearner(wlearner);

                // line-search
                const auto epsilon = epsilon2<scalar_t>();
                const auto x0 = vector_t{vector_t::Zero(loss_tr.size())};
                const auto state = solver->minimize(100, epsilon, loss_tr, x0);
                const auto step = state.x(0);

                wlearner.scale(step);
                wlearner.scale(shrinkage);
                wlearners.push_back(wlearner);

                // update current predictions
                loss_tr.add_wlearner(wlearner);
                loss_vd.add_wlearner(wlearner);
                loss_te.add_wlearner(wlearner);

                status = update_result(loss_tr, loss_vd, loss_te, timer, round + 1, patience, result);

                log_info()
                        << std::setprecision(4)
                        << "[" << (round + 1) << "/" << rounds
                        << "]:tr=" << result.last().m_train
                        << ",vd=" << result.last().m_valid << "|" << status
                        << ",te=" << result.last().m_test
                        << std::setprecision(2)
                        << "," << wlearner
                        << std::setprecision(4)
                        << ",solver=(" << state.m_status << ",i=" << state.m_iterations
                        << ",x=" << state.x(0)
                        << ",f=" << state.f << ",g=" << state.convergence_criteria() << ").";
        }

        // keep only the weak learners up to optimum epoch (on the validation dataset)
        wlearners.erase(
                wlearners.begin() + result.optimum().m_epoch,
                wlearners.end());

        return std::make_pair(result, wlearners);
}

template <typename tloss, typename tweak_learner = typename tloss::weak_learner_t>
static auto train(const task_t& task, const size_t fold, const loss_t& loss, const tuner_t& tuner,
        const string_t& solver_id, const int rounds, const int patience)
{
        training_t result;
        std::vector<tweak_learner> wlearners;
        for (const auto& config : tuner.get(tuner.n_configs()))
        {
                const auto config_result = train_config<tloss>(
                        task, fold, loss, config, solver_id, rounds, patience);

                if (config_result.first < result)
                {
                        result = config_result.first;
                        wlearners = config_result.second;
                }
        }

        log_info() << ">>>" << std::setprecision(4)
                << " tr=" << result.optimum().m_train
                << ",vd=" << result.optimum().m_valid
                << ",te=" << result.optimum().m_test
                << ",round=" << result.optimum().m_epoch
                << "," << result.config() << ".";

        return std::make_pair(result, wlearners);
}

template <typename tweak_learner>
void model_gboost_t<tweak_learner>::to_json(json_t& json) const
{
        nano::to_json(json,
                "rounds", m_rounds,
                "patience", m_patience,
                "solver", m_solver,
                "cumloss", to_string(m_cumloss) + join(enum_values<cumloss>()),
                "shrinkage", to_string(m_shrinkage) + join(enum_values<shrinkage>()),
                "subsampling", to_string(m_subsampling) + join(enum_values<subsampling>()));
}

template <typename tweak_learner>
void model_gboost_t<tweak_learner>::from_json(const json_t& json)
{
        nano::from_json(json,
                "rounds", m_rounds,
                "patience", m_patience,
                "solver", m_solver,
                "cumloss", m_cumloss,
                "shrinkage", m_shrinkage,
                "subsampling", m_subsampling);
}

template <typename tweak_learner>
training_t model_gboost_t<tweak_learner>::train(const task_t& task, const size_t fold, const loss_t& loss)
{
        m_idims = task.idims();
        m_odims = task.odims();

        tuner_t tuner;

        switch (m_shrinkage)
        {
        case shrinkage::off:
                tuner.add_finite("shrinkage", 1.0);
                break;

        case shrinkage::on:
                tuner.add_finite("shrinkage", 0.1, 0.2, 0.5, 1.0);
                break;
        }

        switch (m_subsampling)
        {
        case subsampling::off:
                tuner.add_finite("subsampling", 100);
                break;

        case subsampling::on:
                tuner.add_finite("subsampling", 10, 20, 50, 100);
                break;
        }

        training_t result;
        switch (m_cumloss)
        {
        case cumloss::variance:
                tuner.add_pow10s("lambda", 0.0, -6, +6);
                std::tie(result, m_wlearners) = ::train<gboost_loss_var_t<tweak_learner>>(
                        task, fold, loss, tuner, m_solver, m_rounds, m_patience);
                break;

        case cumloss::average:
        default:
                tuner.add_finite("lambda", 1.0);
                std::tie(result, m_wlearners) = ::train<gboost_loss_avg_t<tweak_learner>>(
                        task, fold, loss, tuner, m_solver, m_rounds, m_patience);
                break;
        }

        return result;
}

template <typename tweak_learner>
tensor3d_t model_gboost_t<tweak_learner>::output(const tensor3d_t& input) const
{
        assert(input.dims() == m_idims);

        tensor3d_t output(m_odims);
        output.zero();

        for (const auto& wlearner : m_wlearners)
        {
                output.vector() += wlearner.output(input).vector();
        }

        return output;
}

template <typename tweak_learner>
bool model_gboost_t<tweak_learner>::save(obstream_t& stream) const
{
        const auto vmajor = static_cast<uint8_t>(major_version);
        const auto vminor = static_cast<uint8_t>(minor_version);

        if (    !stream.write(vmajor) ||
                !stream.write(vminor) ||
                !stream.write(m_idims) ||
                !stream.write(m_odims) ||
                !stream.write(m_rounds) ||
                !stream.write(m_cumloss) ||
                !stream.write(m_shrinkage) ||
                !stream.write(m_subsampling) ||
                !stream.write(m_wlearners.size()))
        {
                return false;
        }

        return  std::all_of(m_wlearners.begin(), m_wlearners.end(),
                [&] (const auto& wlearner) { return wlearner.save(stream); });
}

template <typename tweak_learner>
bool model_gboost_t<tweak_learner>::load(ibstream_t& stream)
{
        uint8_t vmajor = 0x00;
        uint8_t vminor = 0x00;
        size_t n_wlearners = 0;

        if (    !stream.read(vmajor) ||
                !stream.read(vminor) ||
                !stream.read(m_idims) ||
                !stream.read(m_odims) ||
                !stream.read(m_rounds) ||
                !stream.read(m_cumloss) ||
                !stream.read(m_shrinkage) ||
                !stream.read(m_subsampling) ||
                !stream.read(n_wlearners))
        {
                return false;
        }

        m_wlearners.resize(n_wlearners);
        for (auto& wlearner : m_wlearners)
        {
                if (!wlearner.load(stream))
                {
                        return false;
                }
        }

        // todo: more verbose loading (#weak learners, feature or coefficient statistics, idims...)

        return true;
}

template class nano::model_gboost_t<nano::wlearner_linear_t>;
template class nano::model_gboost_t<nano::wlearner_real_stump_t>;
template class nano::model_gboost_t<nano::wlearner_real_table_t>;
template class nano::model_gboost_t<nano::wlearner_discrete_stump_t>;
template class nano::model_gboost_t<nano::wlearner_discrete_table_t>;
