#include "criterion.h"
#include "task.h"
#include "loss.h"
#include <cassert>

namespace nano
{
        criterion_t::criterion_t(const string_t& configuration) :
                clonable_t<criterion_t>(configuration),
                m_lambda(0.0),
                m_type(type::value)
        {
        }

        criterion_t& criterion_t::reset(const rmodel_t& rmodel)
        {
                assert(rmodel);
                return reset(*rmodel);
        }

        criterion_t& criterion_t::reset(const model_t& model)
        {
                m_model = model.clone();
                m_model->save_params(m_params);

                return reset();
        }

        criterion_t& criterion_t::reset(const vector_t& params)
        {
                assert(m_model->psize() == params.size());

                m_model->load_params(params);
                m_params = params;

                return reset();
        }

        criterion_t& criterion_t::reset(const type t)
        {
                m_type = t;

                return reset();
        }

        criterion_t& criterion_t::reset(const scalar_t lambda)
        {
                m_lambda = lambda;

                return reset();
        }

        criterion_t& criterion_t::reset()
        {
                m_vstats.clear();
                m_estats.clear();

                clear();

                return *this;
        }

        void criterion_t::update(const task_t& task, const fold_t& fold, const loss_t& loss)
        {
                update(task, fold, 0, task.n_samples(fold), loss);
        }

        void criterion_t::update(const task_t& task, const fold_t& fold, const size_t begin, const size_t end, const loss_t& loss)
        {
                for (size_t index = begin; index < end; ++ index)
                {
                        update(task.input(fold, index), task.target(fold, index), loss);
                }
        }

        void criterion_t::update(const tensor3d_t& input, const vector_t& target, const loss_t& loss)
        {
                assert(input.size<0>() == m_model->idims());
                assert(input.size<1>() == m_model->irows());
                assert(input.size<2>() == m_model->icols());

                const vector_t& output = m_model->output(input).vector();

                assert(output.size() == m_model->osize());
                assert(target.size() == m_model->osize());

                accumulate(output, target, loss);
        }

        void criterion_t::accumulate(const vector_t& output, const vector_t& target, const loss_t& loss)
        {
                const scalar_t value = loss.value(target, output);
                const scalar_t error = loss.error(target, output);

                m_vstats(value);
                m_estats(error);

                switch (m_type)
                {
                case type::value:
                        accumulate(value);
                        break;

                case type::vgrad:
                        accumulate(m_model->gparam(loss.vgrad(target, output)), value);
                        break;
                }
        }

        criterion_t& criterion_t::update(const criterion_t& other)
        {
                m_vstats(other.m_vstats);
                m_estats(other.m_estats);

                accumulate(other);

                return *this;
        }

        const stats_t<scalar_t>& criterion_t::vstats() const
        {
                assert(m_vstats.count() > 0);
                return m_vstats;
        }

        const stats_t<scalar_t>& criterion_t::estats() const
        {
                assert(m_estats.count() > 0);
                return m_estats;
        }

        size_t criterion_t::count() const
        {
                assert(m_vstats.count() == m_estats.count());
                return m_estats.count();
        }

        const vector_t& criterion_t::params() const
        {
                return m_params;
        }

        tensor_size_t criterion_t::psize() const
        {
                return params().size();
        }

        scalar_t criterion_t::lambda() const
        {
                return m_lambda;
        }
}

