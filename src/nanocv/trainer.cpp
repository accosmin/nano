#include "trainer.h"
#include "common/thread_loop.hpp"
#include "common/timer.h"
#include "common/random.hpp"
#include "common/usampler.hpp"
#include "common/logger.h"
#include "optimize/opt_gd.hpp"
#include "optimize/opt_cgd.hpp"
#include "optimize/opt_lbfgs.hpp"
#include "text.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        trainer_state_t::trainer_state_t(size_t n_parameters)
                :       m_params(n_parameters),
                        m_tvalue(std::numeric_limits<scalar_t>::max()),
                        m_terror(std::numeric_limits<scalar_t>::max()),
                        m_vvalue(std::numeric_limits<scalar_t>::max()),
                        m_verror(std::numeric_limits<scalar_t>::max())
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool trainer_state_t::update(const vector_t& params,
                    scalar_t tvalue, scalar_t terror,
                    scalar_t vvalue, scalar_t verror)
        {
                if (verror < m_verror)
                {
                        m_params = params;
                        m_tvalue = tvalue;
                        m_terror = terror;
                        m_vvalue = vvalue;
                        m_verror = verror;
                        return true;
                }

                else
                {
                        return false;
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool trainer_state_t::update(const trainer_state_t& state)
        {
                return update(state.m_params, state.m_tvalue, state.m_terror, state.m_vvalue, state.m_verror);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        trainer_data_t::trainer_data_t(type t)
                :       m_type(t)
        {
                clear();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        trainer_data_t::trainer_data_t(const model_t& model, type t)
                :       m_type(t)
        {
                clear(model);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::clear(const model_t& model)
        {
                m_model = model.clone();
                m_vgrad.resize(m_model->n_parameters());

                clear();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::clear()
        {
                m_value = 0.0;
                m_error = 0.0;
                m_count = 0;
                m_vgrad.setZero();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::clear(const vector_t& x)
        {
                assert(m_model);

                m_model->load_params(x);
                clear();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::clear(type t)
        {
                m_type = t;
                clear();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::update(const task_t& task, const sample_t& sample, const loss_t& loss)
        {
                assert(m_model);

                const image_t& image = task.image(sample.m_index);
                const vector_t target = image.make_target(sample.m_region);
                assert(image.has_target(target));

                const vector_t output = m_model->value(image, sample.m_region);
                if (m_type == type::vgrad)
                {
                        m_vgrad.noalias() += m_model->gradient(loss.vgrad(target, output));
                }

                m_value += loss.value(target, output);
                m_error += loss.error(target, output);
                m_count ++;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::update(const tensor_t& input, const vector_t& target, const loss_t& loss)
        {
                assert(m_model);

                const vector_t output = m_model->value(input);
                if (m_type == type::vgrad)
                {
                        m_vgrad.noalias() += m_model->gradient(loss.vgrad(target, output));
                }

                m_value += loss.value(target, output);
                m_error += loss.error(target, output);
                m_count ++;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::update_st(const task_t& task, const samples_t& samples, const loss_t& loss)
        {
                for (size_t i = 0; i < samples.size(); i ++)
                {
                        update(task, samples[i], loss);
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::update_st(const tensors_t& inputs, const vectors_t& targets, const loss_t& loss)
        {
                for (size_t i = 0; i < inputs.size(); i ++)
                {
                        update(inputs[i], targets[i], loss);
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::update_mt(const task_t& task, const samples_t& samples, const loss_t& loss, size_t nthreads)
        {
                thread_loop_cumulate<trainer_data_t>
                (
                        samples.size(),
                        [&] (trainer_data_t& data)
                        {
                                assert(m_model);
                                data.clear(m_type);
                                data.clear(*m_model);
                        },
                        [&] (size_t i, trainer_data_t& data)
                        {
                                data.update(task, samples[i], loss);
                        },
                        [&] (trainer_data_t& data)
                        {
                                this->operator +=(data);
                        },
                        nthreads
                );
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::update_mt(const tensors_t& inputs, const vectors_t& targets, const loss_t& loss, size_t nthreads)
        {
                thread_loop_cumulate<trainer_data_t>
                (
                        inputs.size(),
                        [&] (trainer_data_t& data)
                        {
                                assert(m_model);
                                data.clear(m_type);
                                data.clear(*m_model);
                        },
                        [&] (size_t i, trainer_data_t& data)
                        {
                                data.update(inputs[i], targets[i], loss);
                        },
                        [&] (trainer_data_t& data)
                        {
                                this->operator +=(data);
                        },
                        nthreads
                );
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        trainer_data_t& trainer_data_t::operator+=(const trainer_data_t& other)
        {
                m_value += other.m_value;
                m_error += other.m_error;
                m_vgrad += other.m_vgrad;
                m_count += other.m_count;
                return *this;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool trainer_t::train(
                const task_t& task, const samples_t& tsamples, const samples_t& vsamples, size_t batchsize, size_t nthreads,
                const loss_t& loss, scalar_t l2_weight, const string_t& optimizer, size_t iterations, scalar_t epsilon,
                const model_t& model, trainer_state_t& state)
        {
                samples_t utsamples = tsamples;
                samples_t uvsamples = vsamples;

                const scalar_t l2w = l2_weight / model.n_parameters() * (batchsize == 0 ? tsamples.size() : batchsize);

                // construct the optimization problem
                const timer_t timer;

                trainer_data_t ldata(model, trainer_data_t::type::value);
                trainer_data_t gdata(model, trainer_data_t::type::vgrad);

                auto fn_size = [&] ()
                {
                        return ldata.n_parameters();
                };

                auto fn_fval = [&] (const vector_t& x)
                {
                        // training samples: loss value
                        ldata.clear(x);
                        ldata.update_mt(task, tsamples, loss, nthreads);
                        const scalar_t tvalue = ldata.value() + 0.5 * l2w * x.squaredNorm();

                        return tvalue;
                };

                auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        if (batchsize > 0)
                        {
                                // stochastic mode: resample training & validation samples
                                utsamples = ncv::uniform_sample(tsamples, batchsize, random_t<size_t>(0, tsamples.size()));
                                uvsamples = ncv::uniform_sample(vsamples, batchsize, random_t<size_t>(0, vsamples.size()));
                        }

                        // training samples: loss value & gradient
                        gdata.clear(x);
                        gdata.update_mt(task, utsamples, loss, nthreads);
                        const scalar_t tvalue = gdata.value() + 0.5 * l2w * x.squaredNorm();
                        const scalar_t terror = gdata.error();
                        gx = gdata.vgrad() + l2w * x;

                        // validation samples: loss value
                        ldata.clear(x);
                        ldata.update_mt(task, uvsamples, loss, nthreads);
                        const scalar_t vvalue = ldata.value();
                        const scalar_t verror = ldata.error();

                        // update the optimum state
                        state.update(x, tvalue, terror, vvalue, verror);

                        return tvalue;
                };

                auto fn_wlog = [] (const string_t& message)
                {
                        log_warning() << message;
                };
                auto fn_elog = [] (const string_t& message)
                {
                        log_error() << message;
                };
                auto fn_ulog = [&] (const opt_state_t& result, const timer_t& timer)
                {
                        const scalar_t tvalue = state.m_tvalue;
                        const scalar_t terror = state.m_terror;
                        const scalar_t vvalue = state.m_vvalue;
                        const scalar_t verror = state.m_verror;

                        log_info() << "[loss = " << result.f
                                   << ", grad = " << result.g.lpNorm<Eigen::Infinity>()
                                   << ", funs = " << result.n_fval_calls() << "/" << result.n_grad_calls()
                                   << ", L2 = " << l2_weight
                                   << ", train* = " << tvalue << "/" << terror
                                   << ", valid* = " << vvalue << "/" << verror
                                   << "] done in " << timer.elapsed() << ".";
                };

                // assembly optimization problem & optimize the model
                const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);

                const vector_t x = model.params();

                const opt_opulog_t fn_ulog_ref = std::bind(fn_ulog, _1, std::ref(timer));

                if (text::iequals(optimizer, "lbfgs"))
                {
                        optimize::lbfgs(problem, x, iterations, epsilon, fn_wlog, fn_elog, fn_ulog_ref);
                }
                else if (text::iequals(optimizer, "cgd"))
                {
                        optimize::cgd_hs(problem, x, iterations, epsilon, fn_wlog, fn_elog, fn_ulog_ref);
                }
                else if (text::iequals(optimizer, "gd"))
                {
                        optimize::gd(problem, x, iterations, epsilon, fn_wlog, fn_elog, fn_ulog_ref);
                }
                else
                {
                        log_error() << "batch trainer: invalid optimization method <" << optimizer << ">!";
                        return false;
                }

                // OK
                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
	
