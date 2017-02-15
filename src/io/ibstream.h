#pragma once

#include "arch.h"
#include <string>
#include <fstream>
#include <type_traits>

namespace nano
{
        ///
        /// \brief wrapper over binary std::ifstream to serialize particular entities:
        ///     e.g. vectors, matrices, strings, tensors, PODs.
        ///
        class NANO_PUBLIC ibstream_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit ibstream_t(const std::string& path);

                ///
                /// \brief read a POD structure
                ///
                template <typename tstruct, typename = typename std::enable_if<std::is_pod<tstruct>::value>::type>
                bool read(tstruct& pod);

                ///
                /// \brief read a string
                ///
                bool read(std::string& str);

                ///
                /// \brief read given number of bytes
                /// \return the number of bytes actually read
                ///
                std::streamsize read(char* bytes, const std::streamsize num_bytes);

                ///
                /// \brief read a 1D vector
                ///
                template <typename tvector>
                bool read_vector(tvector&);

                ///
                /// \brief read a 2D matrix
                ///
                template <typename tmatrix>
                bool read_matrix(tmatrix&);

                ///
                /// \brief read a ND tensor
                ///
                template <typename ttensor>
                bool read_tensor(ttensor&);

        private:

                template <typename ttensor, typename... tdims>
                static bool resize(ttensor& t, const tdims... dims)
                {
                        t.resize(dims...);
                        return true;
                }

                template <typename ttensor>
                static std::streamsize tsize(const ttensor& t)
                {
                        return t.size() * static_cast<std::streamsize>(sizeof(typename ttensor::Scalar));
                }

        private:

                // attributes
                std::ifstream   m_stream;
        };

        template <typename tstruct, typename>
        bool ibstream_t::read(tstruct& pod)
        {
                const auto size = static_cast<std::streamsize>(sizeof(pod));
                return read(reinterpret_cast<char*>(&pod), size) == size;
        }

        template <typename tvector>
        bool ibstream_t::read_vector(tvector& v)
        {
                typename tvector::Index size = 0;
                return  read(size) &&
                        resize(v, size) &&
                        read(reinterpret_cast<char*>(v.data()), tsize(v)) == tsize(v);
        }

        template <typename tmatrix>
        bool ibstream_t::read_matrix(tmatrix& m)
        {
                typename tmatrix::Index rows = 0, cols = 0;
                return  read(rows) &&
                        read(cols) &&
                        resize(m, rows, cols) &&
                        read(reinterpret_cast<char*>(m.data()), tsize(m)) == tsize(m);
        }

        template <typename ttensor>
        bool ibstream_t::read_tensor(ttensor& t)
        {
                typename ttensor::tdims dims;
                return  read(dims) &&
                        resize(t, dims) &&
                        read(reinterpret_cast<char*>(t.data()), tsize(t)) == tsize(t);
        }
}
