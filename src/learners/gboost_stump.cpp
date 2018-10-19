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
                "stump", m_stype, "stumps", join(enum_values<stump>()),
                "regularization", m_rtype, "regularizations", join(enum_values<regularization>()));
}

void gboost_stump_t::from_json(const json_t& json)
{
        nano::from_json(json,
                "rounds", m_rounds,
                "stump", m_stype,
                "regularization", m_rtype);
}

trainer_result_t gboost_stump_t::train(const task_t& task, const size_t fold, const loss_t& loss)
{
        m_idims = task.idims();
        m_odims = task.odims();

        m_stumps.clear();

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

        log_info() << "gboost-stump: " << 0 << "/" << m_rounds
                << ":tr=" << values_train.avg() << "/" << errors_train.avg()
                << ",vd=" << values_valid.avg() << "/" << errors_valid.avg()
                << ",te=" << values_test.avg() << "/" << errors_test.avg() << std::endl;

        tensor4d_t residuals_train(cat_dims(task.size(fold_train), m_odims));
        tensor3d_t residuals_pos(m_odims), residuals_neg(m_odims);

        for (auto round = 0; round < m_rounds; ++ round)
        {
                for (size_t i = 0, size = task.size(fold_train); i < size; ++ i)
                {
                        const auto input = task.input(fold_train, i);
                        const auto target = task.target(fold_train, i);
                        const auto output = outputs_train.tensor(i);

                        residuals_train.tensor(i).vector() = -loss.vgrad(target, output).vector();
                }

                int best_feature = 0;
                scalar_t best_feature_value = std::numeric_limits<scalar_t>::max();
                stump_t best_stump;

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

                        for (size_t t = 1; t + 1 < thresholds.size(); ++ t)
                        {
                                const auto threshold = thresholds[t];

                                residuals_pos.zero();
                                residuals_neg.zero();

                                for (size_t i = 0, size = task.size(fold_train); i < size; ++ i)
                                {
                                        (fvalues[i] < threshold ? residuals_neg : residuals_pos).vector() +=
                                        residuals_train.tensor(i).vector();
                                }

                                residuals_pos.vector() /= task.size(fold_train);
                                residuals_neg.vector() /= task.size(fold_train);

                                // todo: fit both real and discrete stumps
                        }
                }

                // todo: line-search
                // todo: update current outputs

                ::measure(task, fold_train, outputs_train, loss, errors_train, values_train);
                ::measure(task, fold_valid, outputs_valid, loss, errors_valid, values_valid);
                ::measure(task, fold_test, outputs_test, loss, errors_test, values_test);

                log_info() << "gboost-stump: " << (round + 1) << "/" << m_rounds
                        << ":tr=" << values_train.avg() << "/" << errors_train.avg()
                        << ",vd=" << values_valid.avg() << "/" << errors_valid.avg()
                        << ",te=" << values_test.avg() << "/" << errors_test.avg() << std::endl;
        }

        // todo
        trainer_result_t result;
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
                odata.array() += stump.m_outputs.tensor(oindex).array();
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
