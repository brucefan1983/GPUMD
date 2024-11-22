"""
This script calculates the radial distribution function (RDF) using the OVITO library.
Usage: python rdf_calculator_ovito.py exyzfile cutoff bins
Parameters:
    exyzfile : The path to the input exyz file.
    cutoff   : The cutoff distance for the coordination analysis.
    bins     : The number of bins for the RDF histogram.
Output:
    The calculated RDF values are exported to a text file named "rdf.txt".
Author: Zihan YAN
Date: Aug 20, 2024
"""

import sys
from ovito.io import import_file, export_file
from ovito.modifiers import CoordinationAnalysisModifier,TimeAveragingModifier

# Check if the number of variables is three
if len(sys.argv) != 4:
    print("Error: Invalid number of arguments.")
    print("Usage: python rdf_calculator_ovito.py exyzfile cutoff bins")
    sys.exit(1)

exyzfile = sys.argv[1]
cutoff = sys.argv[2]
bins = sys.argv[3]

pipeline = import_file(exyzfile)
modifier = CoordinationAnalysisModifier(cutoff=cutoff,number_of_bins=bins,partial=True)
pipeline.modifiers.append(modifier)
pipeline.modifiers.append(TimeAveragingModifier(operate_on='table:coordination-rdf'))
export_file(pipeline,"rdf.txt","txt/table",key="coordination-rdf[average]")
