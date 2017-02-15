#pragma once

#include "arch.h"
#include <string>
#include <fstream>
#include <type_traits>

namespace nano
{
        ///
        /// \brief wrapper over binary std::ofstream to serialize particular entities:
        ///     e.g. vectors, matrices, strings, tensors, PODs.
        ///
        class NANO_PUBLIC obstream_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit obstream_t(const std::string& path);

                ///
                /// \brief write a POD structure
                ///
                template <typename tstruct, typename = typename std::enable_if<std::is_pod<tstruct>::value>::type>
                bool write(const tstruct& pod);

                ///
                /// \brief write a string
                ///
                bool write(const std::string& str);

                ///
                /// \brief write given number of bytes
                ///
                bool write(const char* bytes, const std::streamsize num_bytes);

                ///
                /// \brief write a 1D vector
                ///
                template <typename tvector>
                bool write_vector(const tvector&);

                ///
                /// \brief write a 2D matrix
                ///
                template <typename tmatrix>
                bool write_matrix(const tmatrix&);

                ///
                /// \brief write a ND tensor
                ///
                template <typename ttensor>
                bool write_tensor(const ttensor&);

        private:

                template <typename ttensor>
                static std::streamsize tsize(const ttensor& t)
                {
                        return t.size() * static_cast<std::streamsize>(sizeof(typename ttensor::Scalar));
                }

        private:

                // attributes
                std::ofstream   m_stream;
        };

        template <typename tstruct, typename>
        bool obstream_t::write(const tstruct& pod)
        {
                const auto size = static_cast<std::streamsize>(sizeof(pod));
                return write(reinterpret_cast<const char*>(&pod), size);
        }

        template <typename tvector>
        bool obstream_t::write_vector(const tvector& v)
        {
                return  write(v.size()) &&
                        write(reinterpret_cast<const char*>(v.data()), tsize(v));
        }

        template <typename tmatrix>
        bool obstream_t::write_matrix(const tmatrix& m)
        {
                return  write(m.rows()) &&
                        write(m.cols()) &&
                        write(reinterpret_cast<const char*>(m.data()), tsize(m));
        }

        template <typename ttensor>
        bool obstream_t::write_tensor(const ttensor& t)
        {
                return  write(t.dims()) &&
                        write(reinterpret_cast<const char*>(t.data()), tsize(t));
        }
}
