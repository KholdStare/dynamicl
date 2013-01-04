import os
import platform

env = Environment()  # Initialize the environment
env.Append(CPPFLAGS = [ '-O3', '-Wall', '-Werror', '-std=c++0x' ])
env.Append(LIBS = [ 'pthread',
            'boost_unit_test_framework',
            'OpenCL',
            'vigraimpex' ])

# debugging flags
env.Append(CPPFLAGS =  [ '-g', '-pg' ])
env.Append(LINKFLAGS = [ '-g', '-pg' ])
	
bits = 'x86_64'
if platform.machine() != bits:
    bits = 'x86'

if os.environ['AMDAPPSDKROOT']:
    sdkroot = os.environ['AMDAPPSDKROOT']
    env.Append(CPPPATH = [ sdkroot + '/include' ])
    env.Append(LIBPATH = [ sdkroot + '/lib/' + bits ])

commonSource = ['utils.cpp', 'cl_utils.cpp']
mainSource = ['expocl.cpp']
testSource = ['test_suite.cpp']

mainSource.extend(commonSource)
testSource.extend(commonSource)

env.Program(target = 'expocl', source = mainSource)
env.Program(target = 'test_suite', source = testSource)
