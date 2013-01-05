#ifndef KERNEL_H_VY5VUTNS
#define KERNEL_H_VY5VUTNS

#include "cl_common.h"

namespace DynamiCL

{
    /**
     * Represents an OpenCL kernel of particular program,
     * with some auxiliary information to help composition.
     *
     * Allows easy instantiations of kernels for multiple uses.
     */
    struct Kernel
    {
        enum class Range
        {
            SOURCE,
            DESTINATION
        };

        cl::Program const& program;
        char const* name;
        Range const range;

        /**
         * Instantiate a new kernel with the given arguments
         */
        template <typename... Ts>
        cl::Kernel build(Ts&&... args) const
        {
            cl::Kernel kernel(program, name);
            build_impl(kernel, 0, std::forward<Ts>(args)...);
            return kernel;
        }

    private:
        template <typename T, typename... Ts>
        static void build_impl(cl::Kernel& kernel, size_t argIndex, T&& arg, Ts&&... rest)
        {
            kernel.setArg(argIndex, std::forward<T>(arg));
            build_impl(kernel, argIndex+1, std::forward<Ts>(rest)...);
        }

        // no arguments remaining
        static void build_impl(cl::Kernel&, size_t) { }
    };

}

#endif /* end of include guard: KERNEL_H_VY5VUTNS */
