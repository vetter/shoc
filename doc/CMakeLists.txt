# SHOC Documentation

if(SHOC_BUILD_DOC)
    find_program(PDFLATEX pdflatex)
    find_program(BIBTEX bibtex)
    find_program(LATEXMK latexmk)

    if(PDFLATEX AND BIBTEX AND LATEXMK)
        # TODO: Add custom commands to build documentation
        message(STATUS "Documentation build tools found")
    else()
        message(WARNING "Documentation build tools not found (pdflatex, bibtex, latexmk)")
    endif()
else()
    # Just install existing PDF if it exists
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/shoc-manual.pdf")
        install(FILES shoc-manual.pdf
            DESTINATION doc
        )
    endif()
endif()
