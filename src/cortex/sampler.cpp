#include "sampler.h"
#include "math/usampling.hpp"
#include <algorithm>

namespace cortex
{
        template
        <
                typename toperator
        >
        samples_t filter(samples_t samples, const toperator& op)
        {
                samples.erase(std::remove_if(samples.begin(), samples.end(), op), samples.end());
                return samples;
        }

        sampler_t::state_t::state_t(const samples_t& samples, const size_t batchsize)
                :       m_samples(samples),
                        m_batchsize(batchsize)
        {
                // may improve caching!
                std::sort(m_samples.begin(), m_samples.end());
        }

        sampler_t::sampler_t(const samples_t& samples)
        {
                m_states.emplace_back(samples);
        }

        sampler_t::state_t& sampler_t::current()
        {
                assert(!m_states.empty());
                return *m_states.rbegin();
        }

        const sampler_t::state_t& sampler_t::current() const
        {
                assert(!m_states.empty());
                return *m_states.rbegin();
        }

        sampler_t& sampler_t::push(const fold_t fold)
        {
                const auto op = [=] (const sample_t& sample) { return sample.m_fold != fold; };

                m_states.emplace_back(filter(current().m_samples, op));
                return *this;
        }

        sampler_t& sampler_t::push(const protocol p)
        {
                const auto op = [=] (const sample_t& sample) { return sample.m_fold.second != p; };

                m_states.emplace_back(filter(current().m_samples, op));
                return *this;
        }

        sampler_t& sampler_t::push(const annotation a)
        {
                const auto annotated = a == annotation::annotated;
                const auto op = [=] (const sample_t& sample) { return sample.annotated() != annotated; };

                m_states.emplace_back(filter(current().m_samples, op));
                return *this;
        }

        sampler_t& sampler_t::push(const string_t& label)
        {
                const auto op = [=] (const sample_t& sample) { return sample.m_label != label; };

                m_states.emplace_back(filter(current().m_samples, op));
                return *this;
        }

        sampler_t& sampler_t::push(const size_t batchsize)
        {
                m_states.emplace_back(current().m_samples, batchsize);
                return *this;
        }

        sampler_t& sampler_t::push(const samples_t& samples)
        {
                m_states.emplace_back(samples);
                return *this;
        }

        bool sampler_t::pop()
        {
                if (m_states.empty())
                {
                        return false;
                }
                else
                {
                        m_states.pop_back();
                        return true;
                }
        }

        sampler_t& sampler_t::split(size_t percentage, sampler_t& other)
        {
                samples_t tsamples, vsamples;
                math::usplit(current().m_samples, percentage, tsamples, vsamples);

                this->push(tsamples);
                other.push(vsamples);

                return *this;
        }

        samples_t sampler_t::get() const
        {
                const auto& crt = current();

                samples_t samples;

                // use all samples
                if (crt.m_batchsize == 0)
                {
                        samples = crt.m_samples;
                }

                // use a random subset of samples
                else
                {
                        samples = math::usample(crt.m_samples, crt.m_batchsize);
                        std::sort(samples.begin(), samples.end());
                }

                return samples;
        }

        size_t sampler_t::size() const
        {
                return current().m_samples.size();
        }

        bool sampler_t::empty() const
        {
                return current().m_samples.empty();
        }
}
