#pragma once

#include "arch.h"
#include <iosfwd>
#include <vector>
#include <type_traits>
#include <eigen3/Eigen/Core>

namespace io
{
        ///
        /// \brief wrapper over binary std::istream
        ///
        class ZOB_PUBLIC ibstream_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit ibstream_t(std::istream& os);

                ///
                /// \brief read a POD structure
                ///
                template
                <
                        typename tstruct,
                        typename = typename std::enable_if<std::is_pod<tstruct>::value>::type
                >
                ibstream_t& read(tstruct& pod)
                {
                        return read_blob(reinterpret_cast<char*>(&pod), sizeof(pod));
                }

                ///
                /// \brief read a string
                ///
                ibstream_t& read(std::string& str);

                ///
                /// \brief read a std::vector
                ///
                template
                <
                        typename tscalar
                >
                ibstream_t& read(std::vector<tscalar>& vector)
                {
                        std::size_t size;
                        read(size);
                        vector.resize(size);
                        return read_blob(reinterpret_cast<char*>(vector.data()),
                                          static_cast<std::size_t>(vector.size()) * sizeof(tscalar));
                }

                ///
                /// \brief read an Eigen vector or matrix
                ///
                template
                <
                        typename tscalar,
                        int trows,
                        int tcols,
                        int toptions
                >
                ibstream_t& read(Eigen::Matrix<tscalar, trows, tcols, toptions>& matrix)
                {
                        typename Eigen::Matrix<tscalar, trows, tcols, toptions>::Index rows, cols;
                        read(rows);
                        read(cols);
                        matrix.resize(rows, cols);
                        return read_blob(reinterpret_cast<char*>(matrix.data()),
                                         static_cast<std::size_t>(matrix.size()) * sizeof(tscalar));
                }

        private:

                ibstream_t& read_blob(char* data, const std::size_t count);

        private:

                // attributes
                std::istream&   m_stream;
        };
}
