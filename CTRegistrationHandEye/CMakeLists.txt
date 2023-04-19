cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(CTRegistrationHandEye)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/CTRegistrationHandEye")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Tara Kemper (VASST Lab), ")
set(EXTENSION_DESCRIPTION "Hand Eye calibration using CT Registration (wenyao's method)")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/CTRegistrationHandEye.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/CTRegistrationHandEye/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(CTRegistrationHandEye)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
