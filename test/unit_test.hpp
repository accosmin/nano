#pragma once

#include "arch.h"
#include <string>
#include <iostream>
#include "math/abs.hpp"
#include <eigen3/Eigen/Core>

std::string module_name;
std::string case_name;
std::size_t n_failures = 0;

#define NANOCV_BEGIN_MODULE(name) \
int main(int, char* []) \
{ \
        module_name = #name; \
        n_failures = 0; \
        std::cout << "running test module [" << module_name << "] ..." << std::endl;

#define NANOCV_CASE(name) \
        case_name = #name;

#define NANOCV_END_MODULE() \
        if (n_failures > 0) \
        { \
                std::cout << "  failed with " << n_failures << " errors!" << std::endl; \
                exit(EXIT_FAILURE); \
        } \
        else \
        { \
                std::cout << "  no errors detected." << std::endl; \
                exit(EXIT_SUCCESS); \
        } \
}

#define NANOCV_HANDLE_CRITICAL(critical) \
        if (critical) \
        { \
                exit(EXIT_FAILURE); \
        }
#define NANOCV_HANDLE_FAILURE() \
        ++ n_failures; \
        std::cout << __FILE__ << ":" << __LINE__ << ": [" << module_name << "/" << case_name

#define NANOCV_EVALUATE(check, critical) \
        if (!(check)) \
        { \
                NANOCV_HANDLE_FAILURE() \
                        << "]: check {" << NANOCV_STRINGIFY(check) << "} failed!" << std::endl; \
                NANOCV_HANDLE_CRITICAL(critical) \
        }
#define NANOCV_CHECK(check) \
        NANOCV_EVALUATE(check, false)
#define NANOCV_REQUIRE(check) \
        NANOCV_EVALUATE(check, true)

#define NANOCV_THROW(call, exception, critical) \
        try \
        { \
                call; \
                NANOCV_HANDLE_FAILURE() \
                        << "]: call {" << NANOCV_STRINGIFY(call) << "} does not throw!" << std::endl; \
                NANOCV_HANDLE_CRITICAL(critical) \
        } \
        catch (exception&) \
        { \
        } \
        catch (...) \
        { \
                NANOCV_HANDLE_FAILURE() \
                        << "]: call {" << NANOCV_STRINGIFY(call) << "} does not throw {" \
                        << NANOCV_STRINGIFY(exception) << "}!" << std::endl; \
                NANOCV_HANDLE_CRITICAL(critical) \
        }
#define NANOCV_CHECK_THROW(call, exception) \
        NANOCV_THROW(call, exception, false)
#define NANOCV_REQUIRE_THROW(call, exception) \
        NANOCV_THROW(call, exception, true)

#define NANOCV_EVALUATE_BINARY_OP(left, right, op, critical) \
        if (!((left) op (right))) \
        { \
                NANOCV_HANDLE_FAILURE() \
                        << "]: check {" << NANOCV_STRINGIFY(left op right) \
                        << "} failed {" << (left) << " " << NANOCV_STRINGIFY(op) << " " << (right) << "}!" << std::endl; \
                NANOCV_HANDLE_CRITICAL(critical) \
        }

#define NANOCV_EVALUATE_EQUAL(left, right, critical) \
        NANOCV_EVALUATE_BINARY_OP(left, right, ==, critical)
#define NANOCV_CHECK_EQUAL(left, right) \
        NANOCV_EVALUATE_EQUAL(left, right, false)
#define NANOCV_REQUIRE_EQUAL(left, right) \
        NANOCV_EVALUATE_EQUAL(left, right, true)

#define NANOCV_EVALUATE_LESS(left, right, critical) \
        NANOCV_EVALUATE_BINARY_OP(left, right, <, critical)
#define NANOCV_CHECK_LESS(left, right) \
        NANOCV_EVALUATE_LESS(left, right, false)
#define NANOCV_REQUIRE_LESS(left, right) \
        NANOCV_EVALUATE_LESS(left, right, true)

#define NANOCV_EVALUATE_LESS_EQUAL(left, right, critical) \
        NANOCV_EVALUATE_BINARY_OP(left, right, <=, critical)
#define NANOCV_CHECK_LESS_EQUAL(left, right) \
        NANOCV_EVALUATE_LESS_EQUAL(left, right, false)
#define NANOCV_REQUIRE_LESS_EQUAL(left, right) \
        NANOCV_EVALUATE_LESS_EQUAL(left, right, true)

#define NANOCV_EVALUATE_GREATER(left, right, critical) \
        NANOCV_EVALUATE_BINARY_OP(left, right, >, critical)
#define NANOCV_CHECK_GREATER(left, right) \
        NANOCV_EVALUATE_GREATER(left, right, false)
#define NANOCV_REQUIRE_GREATER(left, right) \
        NANOCV_EVALUATE_GREATER(left, right, true)

#define NANOCV_EVALUATE_GREATER_EQUAL(left, right, critical) \
        NANOCV_EVALUATE_BINARY_OP(left, right, >=, critical)
#define NANOCV_CHECK_GREATER_EQUAL(left, right) \
        NANOCV_EVALUATE_GREATER_EQUAL(left, right, false)
#define NANOCV_REQUIRE_GREATER_EQUAL(left, right) \
        NANOCV_EVALUATE_GREATER_EQUAL(left, right, true)

#define NANOCV_EVALUATE_CLOSE(left, right, epsilon, critical) \
        NANOCV_EVALUATE_LESS(math::abs(left - right), epsilon, critical)
#define NANOCV_CHECK_CLOSE(left, right, epsilon) \
        NANOCV_EVALUATE_CLOSE(left, right, epsilon, false)
#define NANOCV_REQUIRE_CLOSE(left, right, epsilon) \
        NANOCV_EVALUATE_CLOSE(left, right, epsilon, true)

#define NANOCV_EVALUATE_EIGEN_CLOSE(left, right, epsilon, critical) \
        NANOCV_EVALUATE_LESS(((left - right).array().abs().maxCoeff()), epsilon, critical)
#define NANOCV_CHECK_EIGEN_CLOSE(left, right, epsilon) \
        NANOCV_EVALUATE_EIGEN_CLOSE(left, right, epsilon, false)
#define NANOCV_REQUIRE_EIGEN_CLOSE(left, right, epsilon) \
        NANOCV_EVALUATE_EIGEN_CLOSE(left, right, epsilon, true)
