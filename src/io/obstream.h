#pragma once

#include "arch.h"
#include <iosfwd>
#include <string>
#include <vector>
#include <type_traits>
#include <eigen3/Eigen/Core>

namespace nano
{
        ///
        /// \brief wrapper over binary std::ostream
        ///
        class NANO_PUBLIC obstream_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit obstream_t(std::ostream& os);

                ///
                /// \brief write a POD structure
                ///
                template
                <
                        typename tstruct,
                        typename = typename std::enable_if<std::is_pod<tstruct>::value>::type
                >
                obstream_t& write(const tstruct& pod)
                {
                        return write_blob(reinterpret_cast<const char*>(&pod),
                                          sizeof(pod));
                }

                ///
                /// \brief write a string
                ///
                obstream_t& write(const std::string& str);

                ///
                /// \brief write a std::vector
                ///
                template
                <
                        typename tvalue
                >
                obstream_t& write(const std::vector<tvalue>& vector)
                {
                        write(vector.size());
                        return write_blob(reinterpret_cast<const char*>(vector.data()),
                                          static_cast<std::size_t>(vector.size()) * sizeof(tvalue));
                }

                ///
                /// \brief write an Eigen::Vector
                ///
                template
                <
                        typename tvalue,
                        int trows,
                        int tcols,
                        int toptions
                >
                obstream_t& write(const Eigen::Matrix<tvalue, trows, tcols, toptions>& matrix)
                {
                        write(matrix.rows());
                        write(matrix.cols());
                       return write_blob(reinterpret_cast<const char*>(matrix.data()),
                                         static_cast<std::size_t>(matrix.size()) * sizeof(tvalue));
                }

        private:

                obstream_t& write_blob(const char* data, const std::size_t count);

        private:

                // attributes
                std::ostream&   m_stream;
        };
}
