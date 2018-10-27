#include "gboost.h"
#include "solver.h"
#include "core/logger.h"
#include "gboost_stump.h"
#include "core/ibstream.h"
#include "core/obstream.h"

using namespace nano;

static void measure(const task_t& task, const fold_t& fold, const tensor4d_t& outputs, const loss_t& loss,
        stats_t& errors, stats_t& values)
{
        errors.clear();
        values.clear();

        // todo: use multiple threads to speed up computation
        // todo: extend loss_t to also process mapped 3D tensors (e.g. outputs)
        for (size_t i = 0, size = task.size(fold); i < size; ++ i)
        {
                const auto input = task.input(fold, i);
                const auto target = task.target(fold, i);
                const auto output = outputs.tensor(i);

                errors(loss.error(target, output));
                values(loss.value(target, output));
        }
}

void gboost_stump_t::to_json(json_t& json) const
{
        nano::to_json(json,
                "rounds", m_rounds,
                "patience", m_patience,
                "stump", m_stype, "stumps", join(enum_values<stump>()),
                "solver", m_solver, "solvers", join(get_solvers().ids()),
                "regularization", m_rtype, "regularizations", join(enum_values<regularization>()));
}

void gboost_stump_t::from_json(const json_t& json)
{
        nano::from_json(json,
                "rounds", m_rounds,
                "patience", m_patience,
                "stump", m_stype,
                "solver", m_solver,
                "regularization", m_rtype);
}

trainer_result_t gboost_stump_t::train(const task_t& task, const size_t fold, const loss_t& loss)
{
        m_idims = task.idims();
        m_odims = task.odims();

        m_stumps.clear();

        trainer_result_t result("<config>");
        timer_t timer;

        const auto fold_train = fold_t{fold, protocol::train};
        const auto fold_valid = fold_t{fold, protocol::valid};
        const auto fold_test = fold_t{fold, protocol::test};

        tensor4d_t outputs_train(cat_dims(task.size(fold_train), m_odims));
        tensor4d_t outputs_valid(cat_dims(task.size(fold_valid), m_odims));
        tensor4d_t outputs_test(cat_dims(task.size(fold_test), m_odims));

        outputs_train.zero();
        outputs_valid.zero();
        outputs_test.zero();

        stats_t errors_train, errors_valid, errors_test;
        stats_t values_train, values_valid, values_test;

        ::measure(task, fold_train, outputs_train, loss, errors_train, values_train);
        ::measure(task, fold_valid, outputs_valid, loss, errors_valid, values_valid);
        ::measure(task, fold_test, outputs_test, loss, errors_test, values_test);

        result.update(trainer_state_t{timer.milliseconds(), 0,
                {values_train.avg(), errors_train.avg()},
                {values_valid.avg(), errors_valid.avg()},
                {values_test.avg(), errors_test.avg()}},
                m_patience);

        log_info() << result;

        tensor4d_t residuals_train(cat_dims(task.size(fold_train), m_odims));
        tensor3d_t residuals_pos1(m_odims), residuals_pos2(m_odims);
        tensor3d_t residuals_neg1(m_odims), residuals_neg2(m_odims);

        stump_t stump;
        tensor4d_t stump_outputs_train(cat_dims(task.size(fold_train), m_odims));

        tensor4d_t targets(cat_dims(task.size(fold_train), m_odims));
        for (size_t i = 0, size = task.size(fold_train); i < size; ++ i)
        {
                const auto target = task.target(fold_train, i);
                targets.tensor(i) = target.tensor();
        }

        gboost_lsearch_function_t func(targets, outputs_train, stump_outputs_train, loss);

        const auto solver = get_solvers().get(m_solver);
        // todo: critical error is the solver is not found
        assert(solver != nullptr);

        for (auto round = 0; round < m_rounds && !result.is_done(); ++ round)
        {
                for (size_t i = 0, size = task.size(fold_train); i < size; ++ i)
                {
                        const auto input = task.input(fold_train, i);
                        const auto target = task.target(fold_train, i);
                        const auto output = outputs_train.tensor(i);

                        const auto vgrad = loss.vgrad(target, output);
                        residuals_train.vector(i) = -vgrad.vector();
                }

                scalar_t best_value = std::numeric_limits<scalar_t>::max();

                // todo: generalize this - e.g. to use features that are products of two input features
                for (auto feature = 0; feature < nano::size(m_idims); ++ feature)
                {
                        scalars_t fvalues(task.size(fold_train));
                        for (size_t i = 0, size = task.size(fold_train); i < size; ++ i)
                        {
                                const auto input = task.input(fold_train, i);
                                fvalues[i] = input(feature);
                        }

                        auto thresholds = fvalues;
                        std::sort(thresholds.begin(), thresholds.end());
                        thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());

                        for (size_t t = 0; t + 1 < thresholds.size(); ++ t)
                        {
                                const auto threshold = (thresholds[t + 0] + thresholds[t + 1]) / 2;

                                residuals_pos1.zero(), residuals_pos2.zero();
                                residuals_neg1.zero(), residuals_neg2.zero();

                                int cnt_pos = 0, cnt_neg = 0;
                                for (size_t i = 0, size = task.size(fold_train); i < size; ++ i)
                                {
                                        const auto residual = residuals_train.tensor(i).array();
                                        if (fvalues[i] < threshold)
                                        {
                                                cnt_neg ++;
                                                residuals_neg1.array() += residual;
                                                residuals_neg2.array() += residual * residual;
                                        }
                                        else
                                        {
                                                cnt_pos ++;
                                                residuals_pos1.array() += residual;
                                                residuals_pos2.array() += residual * residual;
                                        }
                                }

                                const auto value =
                                        (residuals_pos2.array().sum() - residuals_pos1.array().square().sum() / cnt_pos) +
                                        (residuals_neg2.array().sum() - residuals_neg1.array().square().sum() / cnt_neg);

                                //log_info() << "feature = " << feature
                                //        << ", threshold = " << threshold
                                //        << ", value = " << value
                                //        << ", count = " << cnt_neg << "+" << cnt_pos << "=" << task.size(fold_train);

                                if (value < best_value)
                                {
                                        best_value = value;
                                        stump.m_feature = feature;
                                        stump.m_threshold = threshold;
                                        stump.m_outputs.resize(cat_dims(2, m_odims));
                                        stump.m_outputs.vector(0) = residuals_neg1.vector() / cnt_neg;
                                        stump.m_outputs.vector(1) = residuals_pos1.vector() / cnt_pos;
                                }

                                // todo: fit both real and discrete stumps
                        }
                }

                // line-search
                for (size_t i = 0, size = task.size(fold_train); i < size; ++ i)
                {
                        const auto input = task.input(fold_train, i);
                        const auto oindex = input(stump.m_feature) < stump.m_threshold ? 0 : 1;
                        stump_outputs_train.tensor(i) = stump.m_outputs.tensor(oindex);
                        log_info() << "i=" << i
                                << ": soutputs = " << stump_outputs_train.vector(i).minCoeff() << "," << stump_outputs_train.vector(i).maxCoeff()
                                << ", soutputs = " << stump.m_outputs.vector(oindex).minCoeff() << "," << stump.m_outputs.vector(oindex).maxCoeff();
                }

                log_info() << "best_value = " << best_value
                        << ", stump0 = " << stump.m_outputs.array(0).minCoeff() << "," << stump.m_outputs.array(0).maxCoeff()
                        << ", stump1 = " << stump.m_outputs.array(1).minCoeff() << "," << stump.m_outputs.array(1).maxCoeff();

                log_info() << "outputs = [" << outputs_train.array().minCoeff() << "," << outputs_train.array().maxCoeff()
                        << "], stump = [" << stump_outputs_train.array().minCoeff() << "," << stump_outputs_train.array().maxCoeff() << "]";

                const auto logger = [] (const solver_state_t& state)
                {
                        log_info() << state;
                        return true;
                };

                const auto state = solver->minimize(100, epsilon2<scalar_t>(), func, vector_t::Constant(1, 0), logger);
                const auto step = state.x(0);

                //log_info() << "step = " << step << ", state.f=" << state.f << ", state.g=" << state.g.transpose();

                stump.m_outputs.vector() *= step;

                // update current outputs
                for (size_t i = 0, size = task.size(fold_train); i < size; ++ i)
                {
                        const auto input = task.input(fold_train, i);
                        const auto oindex = input(stump.m_feature) < stump.m_threshold ? 0 : 1;
                        outputs_train.array(i) += stump.m_outputs.array(oindex);
                }
                for (size_t i = 0, size = task.size(fold_valid); i < size; ++ i)
                {
                        const auto input = task.input(fold_valid, i);
                        const auto oindex = input(stump.m_feature) < stump.m_threshold ? 0 : 1;
                        outputs_valid.array(i) += stump.m_outputs.array(oindex);
                }
                for (size_t i = 0, size = task.size(fold_test); i < size; ++ i)
                {
                        const auto input = task.input(fold_test, i);
                        const auto oindex = input(stump.m_feature) < stump.m_threshold ? 0 : 1;
                        outputs_test.array(i) += stump.m_outputs.array(oindex);
                }

                m_stumps.push_back(stump);

                ::measure(task, fold_train, outputs_train, loss, errors_train, values_train);
                ::measure(task, fold_valid, outputs_valid, loss, errors_valid, values_valid);
                ::measure(task, fold_test, outputs_test, loss, errors_test, values_test);

                result.update(trainer_state_t{timer.milliseconds(), round + 1,
                        {values_train.avg(), errors_train.avg()},
                        {values_valid.avg(), errors_valid.avg()},
                        {values_test.avg(), errors_test.avg()}},
                        m_patience);

                log_info() << result << ",feature=" << stump.m_feature << ",gamma=" << step
                        << ",solver=" << state.m_status
                        << "|i=" << state.m_iterations << "|f=" << state.f << "|g=" << state.convergence_criteria() << ".";
        }

        return result;
}

tensor3d_t gboost_stump_t::output(const tensor3d_t& input) const
{
        assert(input.dims() == m_idims);

        tensor3d_t output(m_odims);
        output.zero();

        const auto idata = input.array();
        auto odata = output.array();

        for (const auto& stump : m_stumps)
        {
                const auto oindex = idata(stump.m_feature) < stump.m_threshold ? 0 : 1;
                odata.array() += stump.m_outputs.array(oindex);
        }

        return output;
}

bool gboost_stump_t::save(obstream_t& stream) const
{
        if (    !stream.write(m_idims) ||
                !stream.write(m_odims) ||
                !stream.write(m_rounds) ||
                !stream.write(m_stype) ||
                !stream.write(m_rtype) ||
                !stream.write(m_stumps.size()))
        {
                return false;
        }

        for (const auto& stump : m_stumps)
        {
                assert(stump.m_feature >= 0 && stump.m_feature < nano::size(m_idims));
                assert(stump.m_outputs.dims() == cat_dims(2, m_odims));

                if (    !stream.write(stump.m_feature) ||
                        !stream.write(stump.m_threshold) ||
                        !stream.write_tensor(stump.m_outputs))
                {
                       return false;
                }
        }

        return true;
}

bool gboost_stump_t::load(ibstream_t& stream)
{
        size_t n_stumps = 0;
        if (    !stream.read(m_idims) ||
                !stream.read(m_odims) ||
                !stream.read(m_rounds) ||
                !stream.read(m_stype) ||
                !stream.read(m_rtype) ||
                !stream.read(n_stumps))
        {
                return false;
        }

        m_stumps.resize(n_stumps);
        for (auto& stump : m_stumps)
        {
                if (    !stream.read(stump.m_feature) ||
                        !stream.read(stump.m_threshold) ||
                        !stream.read_tensor(stump.m_outputs) ||
                        stump.m_feature < 0 ||
                        stump.m_feature >= nano::size(m_idims) ||
                        stump.m_outputs.dims() != cat_dims(2, m_odims))
                {
                        return false;
                }
        }

        // todo: more verbose loading (#stumps, feature or coefficient statistics, idims...)

        return true;
}

probes_t gboost_stump_t::probes() const
{
        // todo: add probes here to measure the training and the evaluation time
        probes_t probes;
        return probes;
}
