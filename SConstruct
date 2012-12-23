import os
import platform

env = Environment()  # Initialize the environment
env['CPPFLAGS'] = '-O3 -g -std=c++0x'
env.Append(LIBS = [ 'pthread',
            'boost_unit_test_framework',
            'OpenCL',
            'vigraimpex' ])
	
bits = 'x86_64'
if platform.machine() != bits:
    bits = 'x86'

if os.environ['AMDAPPSDKROOT']:
    sdkroot = os.environ['AMDAPPSDKROOT']
    env.Append(CPPPATH = [ sdkroot + '/include' ])
    env.Append(LIBPATH = [ sdkroot + '/lib/' + bits ])

env.Program(target = 'expocl', source = ['expocl.cpp', 'utils.cpp'])
env.Program(target = 'test_suite', source = ['test_suite.cpp', 'utils.cpp'])
