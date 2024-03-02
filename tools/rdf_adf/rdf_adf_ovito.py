from ovito.io import *
from ovito.modifiers import *
import numpy as np

def Proc_rdf(indump, outrdf, rdfcf=10.0, bins=100):

    # Load input data.
    pipeline = import_file(indump)
    nfram = pipeline.source.num_frames

    # Calculate partial RDFs with time average:
    pipeline.modifiers.append(
        CoordinationAnalysisModifier(
            cutoff=rdfcf,
            number_of_bins=bins,
            partial=True))
    pipeline.modifiers.append(
        TimeAveragingModifier(
            operate_on='table:coordination-rdf'))
    TimeAveragingModifier.interval=(nfram//2, nfram-1)

    data = pipeline.compute()
    export_file(pipeline, outrdf, 'txt/table', key='coordination-rdf[average]')


def Proc_adf(indump, outadf, del_atom_type=[], adfcf=[], bins=180):

    pipeline = import_file(indump)

    data = pipeline.compute()

    # if you want delete some elements before compute adf.
    if del_atom_type != []:
        for di in del_atom_type:
            pipeline.modifiers.append(
                SelectTypeModifier(
                    operate_on='particles',
                    property="Particle Type",
                    types={di}))
        pipeline.modifiers.append(
            DeleteSelectedModifier())
        data = pipeline.compute()

    # Before calculating the RDF, you need to form a key information
    #   and there are several different ways to generate it

    # mode style ref from:
    # https://www.ovito.org/docs/current/python/modules/ovito_modifiers.html#ovito.modifiers.CreateBondsModifier.mode
    if adfcf == []:                      # mode = "VdWRadius"
        # uses a distance cutoff that is derived from the vdw_radius
        pipeline.modifiers.append(CreateBondsModifier(
                mode=CreateBondsModifier.Mode.VdWRadius,lower_cutoff=0.1))
    elif isinstance(adfcf, (int,float)): # mode = "Uniform"
        # uses a single uniform cutoff distance for creating bonds
        cfs = adfcf
        pipeline.modifiers.append(CreateBondsModifier(
                mode=CreateBondsModifier.Mode.Uniform,cutoff=cfs))
    elif isinstance(adfcf, dict):        # mode = "Pairwise"
        # specify a separate cutoff distance for each pairwise combination of particle types
        cbm = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise,lower_cutoff=0.1)
        for di in adfcf:
            p1, p2 = di.split('-')
            cbm.set_pairwise_cutoff(p1, p2, adfcf[di])
            pipeline.modifiers.append(cbm)
    else:
        raise "error with adfcf types, there are three types:\n num, [], dict"

    # Calculate instantaneous bond angle distribution.
    pipeline.modifiers.append(BondAnalysisModifier(bins = bins))
    # Perform time averaging of the DataTable 'bond-angle-distr'.
    pipeline.modifiers.append(TimeAveragingModifier(operate_on='table:bond-angle-distr'))
    # Compute and export the time-averaged histogram to a text file.
    export_file(pipeline, outadf, 'txt/table', key='bond-angle-distr[average]')


def main():

    # The input file can be any file format that Ovito can read, ref from
    # https://www.ovito.org/docs/current/reference/file_formats/file_formats_input.html#file-formats-input
    indump = "Ex_xyz/water.xyz"    # *.xyz *.exyz *.lammpstrj XDATCAR et. al.

    outrdf = "./rdf.dat"
    cf = 8.0           # cutoff with units Angstrom
    bs = 400          # delt r with units Angstrom
    Proc_rdf(indump, outrdf, rdfcf=cf, bins=bs)

    # cf = Num                  # mode="Uniform"
    # cf = []                   # mode="VdWRadius"
    # cf = {'O-H':1.2}          # mode="Pairwise"
    bs = 360
    outadf = "./adf-dp.dat"

    cf = {'O-O': 3.5}
    outadf = "./adf-ooo.dat"
    Proc_adf(indump, outadf, del_atom_type=['H'], adfcf=cf, bins=bs)

    cf = {'O-H': 1.2}
    outadf = "./adf-hoh.dat"
    Proc_adf(indump, outadf, adfcf=cf, bins=bs)

    #cf = 1.2
    #outadf = "./adf-c-hoh.dat"
    #Proc_adf(indump, outadf, adfcf=cf, bins=bs)

    cf = []
    outadf = "./adf-c1-hoh.dat"
    Proc_adf(indump, outadf, adfcf=cf, bins=bs)

if __name__ == '__main__':
    main()
