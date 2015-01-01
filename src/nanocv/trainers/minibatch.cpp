#include "batch.h"
#include "accumulator.h"
#include "sampler.h"
#include "file/logger.h"
#include "util/log_search.hpp"
#include "util/timer.h"
#include "optimize.h"

namespace ncv
{
        namespace detail
        {        
                static opt_state_t minibatch_train(
                        trainer_data_t& data,
                        batch_optimizer optimizer, size_t iterations, scalar_t epsilon, size_t epoch,
                        timer_t& timer, trainer_result_t& result)
                {
                        size_t iteration = 0;

                        // construct the optimization problem
                        auto fn_size = ncv::make_opsize(data);
                        auto fn_fval = ncv::make_opfval(data);
                        auto fn_grad = ncv::make_opgrad(data);

                        auto fn_wlog = ncv::make_opwlog();
                        auto fn_elog = ncv::make_opelog();
                        auto fn_ulog = [&] (const opt_state_t& state)
                        {
                                if ((++ iteration) == iterations)
                                {
                                        const scalar_t tvalue = data.m_gacc.value();
                                        const scalar_t terror_avg = data.m_gacc.avg_error();
                                        const scalar_t terror_var = data.m_gacc.var_error();

                                        // validation samples: loss value
                                        data.m_lacc.reset(state.x);
                                        data.m_lacc.update(data.m_task, data.m_vsampler.get(), data.m_loss);
                                        const scalar_t vvalue = data.m_lacc.value();
                                        const scalar_t verror_avg = data.m_lacc.avg_error();
                                        const scalar_t verror_var = data.m_lacc.var_error();

                                        // update the optimum state
                                        result.update(state.x, tvalue, terror_avg, terror_var, vvalue, verror_avg, verror_var,
                                                      epoch, scalars_t({ data.m_lacc.lambda() }));

                                        log_info()
                                        << "[train = " << tvalue << "/" << terror_avg << "/=" << data.m_tsampler.size()
                                        << ", valid = " << vvalue << "/" << verror_avg << "/=" << data.m_vsampler.size()
                                        << ", xnorm = " << state.x.lpNorm<Eigen::Infinity>()
                                        << ", gnorm = " << state.g.lpNorm<Eigen::Infinity>()
                                        << ", epoch = " << epoch
                                        << ", lambda = " << data.m_lacc.lambda()
                                        << ", calls = " << state.n_fval_calls() << "/" << state.n_grad_calls()
                                        << "] done in " << timer.elapsed() << ".";
                                }
                        };

                        // assembly optimization problem & optimize the model
                        return ncv::minimize(fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog,
                                             data.m_x0, optimizer, iterations, epsilon);
                }
        }

        trainer_result_t minibatch_train(
                const model_t& model, const task_t& task, sampler_t& tsampler, sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion, size_t batchsize, scalar_t batchratio,
                batch_optimizer optimizer, size_t epochs, size_t iterations, scalar_t epsilon)
        {
                // operator to train for a given regularization factor
                const auto op = [&] (scalar_t lambda)
                {
                        trainer_result_t result;
                        timer_t timer;

                        // optimize the model
                        vector_t x0;
                        model.save_params(x0);

                        for (size_t epoch = 1, tsize = batchsize; epoch <= epochs;
                                epoch ++, tsize = static_cast<size_t>(tsize * batchratio))
                        {
                                // random subset of training samples
                                if (tsize < tsampler.size())
                                {
                                        tsampler.setup(sampler_t::stype::uniform, tsize);
                                }
                                else
                                {
                                        tsampler.setup(sampler_t::stype::batch);
                                }

                                // all validation samples
                                vsampler.setup(sampler_t::stype::batch);

                                accumulator_t lacc(model, nthreads, criterion, criterion_t::type::value, lambda);
                                accumulator_t gacc(model, nthreads, criterion, criterion_t::type::vgrad, lambda);

                                trainer_data_t data(task, tsampler, vsampler, loss, x0, lacc, gacc);

                                const opt_state_t state =
                                detail::minibatch_train(data, optimizer, iterations, epsilon, epoch, timer, result);
                                x0 = state.x;

                                // NB: this will cause resampling of the training data!
                        }

                        // OK
                        return result;
                };

                // tune the regularization factor (if needed)
                if (accumulator_t::can_regularize(criterion))
                {
                        return log10_min_search(op, -4.0, +4.0, 0.5, 4).first;
                }

                else
                {
                        return op(0.0);
                }
        }
}
	
