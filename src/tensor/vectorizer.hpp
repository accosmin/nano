#ifndef NANOCV_VECTORIZER_H
#define NANOCV_VECTORIZER_H

#include "matrix.hpp"

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief serialize data to vector
                ///
                template
                <
                        typename tscalar,
                        typename tsize
                >
                class ovectorizer_t
                {
                public:

                        typedef typename vector_types_t<tscalar>::tvector       tvector;

                        ///
                        /// \brief constructor
                        ///
                        ovectorizer_t(tvector& data)
                                :       m_data(data),
                                        m_pos(0)
                        {
                        }

                        ///
                        /// \brief serialize scalar
                        ///
                        ovectorizer_t& operator<<(tscalar val)
                        {
                                m_data(m_pos ++) = val;
                                m_pos ++;
                                return *this;
                        }

                        ///
                        /// \brief serialize tensor
                        ///
                        template
                        <
                                typename ttensor
                        >
                        ovectorizer_t& operator<<(const ttensor& t)
                        {
                                std::copy(t.data(), t.data() + t.size(), m_data.data() + m_pos);
                                m_pos += t.size();
                                return *this;
                        }

                private:

                        // attributes
                        tvector&        m_data;         ///< serialize to this vector
                        tsize           m_pos;          ///< current streaming position
                };

                ///
                /// \brief serialize vectors of known types
                ///
                template
                <
                        typename tscalar,
                        typename tsize,
                        typename tdata
                >
                ovectorizer_t<tscalar, tsize>& operator<<(ovectorizer_t<tscalar, tsize>& s, const std::vector<tdata>& datas)
                {
                        for (const tdata& data : datas)
                        {
                                s << data;
                        }
                        return s;
                }

                ///
                /// \brief deserialize vector to data
                ///
                template
                <
                        typename tscalar,
                        typename tsize
                >
                class ivectorizer_t
                {
                public:

                        typedef typename vector_types_t<tscalar>::tvector       tvector;

                        ///
                        /// \brief constructor
                        ///
                        ivectorizer_t(const tvector& data)
                                :       m_data(data),
                                        m_pos(0)
                        {
                        }

                        ///
                        /// \brief deserialize scalar
                        ///
                        ivectorizer_t& operator>>(tscalar& val)
                        {
                                val = m_data(m_pos ++);
                                return *this;
                        }

                        ///
                        /// \brief deserialize tensor
                        ///
                        template
                        <
                                typename ttensor
                        >
                        ivectorizer_t& operator>>(ttensor& t)
                        {
                                auto segm = m_data.segment(m_pos, t.size());
                                std::copy(segm.data(), segm.data() + segm.size(), t.data());
                                m_pos += t.size();
                                return *this;
                        }

                private:

                        // attributes
                        const tvector&  m_data;         ///< serialize from this vector
                        tsize           m_pos;          ///< current streaming position
                };

                ///
                /// \brief deserialize vectors of known types
                ///
                template
                <
                        typename tscalar,
                        typename tsize,
                        typename tdata
                >
                ivectorizer_t<tscalar, tsize>& operator>>(ivectorizer_t<tscalar, tsize>& s, std::vector<tdata>& datas)
                {
                        for (tdata& data : datas)
                        {
                                s >> data;
                        }
                        return s;
                }
        }
}

#endif // NANOCV_VECTORIZER_H
