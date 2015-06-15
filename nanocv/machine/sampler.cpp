#include "sampler.h"
#include "task.h"
#include "math/usampling.hpp"
#include <algorithm>

namespace ncv
{
        sampler_t::sampler_t(const task_t& task)
                :       sampler_t(task.samples())
        {
        }

        sampler_t::sampler_t(const samples_t& samples)
                :       m_osamples(samples),
                        m_samples(samples),
                        m_stype(stype::batch),
                        m_ssize(0)
        {
        }

        void sampler_t::reset()
        {
                // collect all available samples (no restriction)
                m_samples = m_osamples;
        }

        sampler_t& sampler_t::setup(fold_t fold)
        {
                m_samples.erase(std::remove_if(m_samples.begin(), m_samples.end(),
                                [&] (const sample_t& sample) { return sample.m_fold != fold; }),
                                m_samples.end());

                return order();
        }

        sampler_t& sampler_t::setup(protocol p)
        {
                m_samples.erase(std::remove_if(m_samples.begin(), m_samples.end(),
                                [&] (const sample_t& sample) { return sample.m_fold.second != p; }),
                                m_samples.end());

                return order();
        }

        sampler_t& sampler_t::setup(stype s, size_t size)
        {
                m_stype = s;
                m_ssize = size;

                return *this;
        }

        sampler_t& sampler_t::setup(atype a)
        {
                const bool annotated = a == atype::annotated;

                m_samples.erase(std::remove_if(m_samples.begin(), m_samples.end(),
                                [&] (const sample_t& sample) { return sample.annotated() != annotated; }),
                                m_samples.end());

                return order();
        }

        sampler_t& sampler_t::setup(const string_t& label)
        {
                m_samples.erase(std::remove_if(m_samples.begin(), m_samples.end(),
                                [&] (const sample_t& sample) { return sample.m_label != label; }),
                                m_samples.end());

                return order();
        }

        sampler_t& sampler_t::split(size_t percentage, sampler_t& other)
        {
                samples_t tsamples, vsamples;
                math::usplit(m_samples, percentage, tsamples, vsamples);

                m_samples = tsamples;
                other.m_samples = vsamples;
                other.order();

                return order();
        }

        samples_t sampler_t::get() const
        {
                samples_t samples;

                switch (m_stype)
                {
                case stype::batch:
                        samples = m_samples;
                        break;

                case stype::uniform:
                        samples = math::usample(m_samples, m_ssize);
                        std::sort(samples.begin(), samples.end());
                        break;
                }

                return samples;
        }

        sampler_t& sampler_t::order()
        {
                std::sort(m_samples.begin(), m_samples.end());

                return *this;
        }
}
