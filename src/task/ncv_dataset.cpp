#include "ncv_dataset.h"
#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/base_object.hpp>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        dataset::dataset()
                :       m_irows(0), m_icols(0), m_tsize(0)
        {
        }

        //-------------------------------------------------------------------------------------------------

        void dataset::clear()
        {
                m_samples.clear();
                m_irows = 0;
                m_icols = 0;
                m_tsize = 0;
        }

        //-------------------------------------------------------------------------------------------------

        bool dataset::add(const matrix_t& input, const vector_t& target, const string_t& label)
        {
                // check input
                if (input.size() == 0)
                {
                        return false;
                }

                if (m_irows == 0 && m_icols == 0)
                {
                        m_irows = input.rows();
                        m_icols = input.cols();
                }

                if (static_cast<size_t>(input.rows()) != m_irows ||
                    static_cast<size_t>(input.cols()) != m_icols)
                {
                        return false;
                }

                if (target.size() > 0)
                {
                        if (m_tsize == 0)
                        {
                                m_tsize = target.size();
                        }

                        if (static_cast<size_t>(target.size()) != m_tsize)
                        {
                                return false;
                        }
                }

                sample s;
                s.m_input = input;
                s.m_target = target;
                s.m_label = label;
                m_samples.push_back(s);
                return true;
        }

        //-------------------------------------------------------------------------------------------------

        bool dataset::add(const matrix_t& input, const vector_t& target)
        {
                return add(input, target, string_t());
        }

        //-------------------------------------------------------------------------------------------------

        bool dataset::add(const matrix_t& input)
        {
                return add(input, vector_t(), string_t());
        }

        //-------------------------------------------------------------------------------------------------

        bool dataset::has_input(index_t s) const
        {
                return  static_cast<size_t>(input(s).rows()) == irows() &&
                        static_cast<size_t>(input(s).cols()) == icols();
        }

        //-------------------------------------------------------------------------------------------------

        bool dataset::has_target(index_t s) const
        {
                return  static_cast<size_t>(target(s).size()) == tsize();
        }

        //-------------------------------------------------------------------------------------------------

        bool dataset::has_label(index_t s) const
        {
                return !label(s).empty();
        }

        //-------------------------------------------------------------------------------------------------
        
        bool dataset::save(const string_t& path) const
        {
                std::ofstream ofs(path, std::ios::binary);
                
                boost::archive::binary_oarchive oa(ofs);
                oa << m_samples;
                oa << m_irows;
                oa << m_icols;
                oa << m_tsize;

                return ofs.good();
        }

        //-------------------------------------------------------------------------------------------------

        bool dataset::load(const string_t& path)
        {
                std::ifstream ifs(path, std::ios::binary);
                
                boost::archive::binary_iarchive ia(ifs);
                ia >> m_samples;
                ia >> m_irows;
                ia >> m_icols;
                ia >> m_tsize;
                
                return ifs.good();
        }

        //-------------------------------------------------------------------------------------------------
}
