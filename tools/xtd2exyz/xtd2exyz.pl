#!perl
#**********************************************************
#*                                                        *
#*     XTD2XYZ - Convert XTD files into XYZ format        *
#*                                                        *
#**********************************************************
# Version: 0.1
# Author: Andrea Minoia
# Date: 08/09/2010
#
# Convert MS trajectory xtd file into XYZ trajectory file.
# Backup of files that are about to be overwritten is managed
# by MS. The most recent file is that with higher index number (N)
# The script has to be in the same directory of the
# structure to modify and the user has to update the
# variable $doc (line 31) according to the name of the
# file containing the trajectory.
# The xmol trajectory is stored in trj.txt file and it is not
# possible to rename the file within MS, nor it is possible to
# automatically export it as xyz or car file. You should manage
# the new trajectory manually for further use (e.g. VMD)
#
# Modificator: Sobereva (sobereva@sina.com)
# Date: 2012-May-23

# Modificator: nanxu (tamas@zju.edu.cn)
# Date: 2022-Jau-03
# Add support for lattice

# Date: 2024-4-25
# Modificator: chenzherui (chenzherui0124@foxmail.com)
# Energy unit eV; Force unit eV/Å
# Materials Studio xtd trajectory to exyz 
# Optimize programs to improve efficienc, fix bugs

use strict;
use MaterialsScript qw(:all);

# Open the multiframe trajectory structure file or die
my $doc = $Documents{"small.xtd"};

die "no document" unless $doc;

my $trajectory = $doc->Trajectory;
my $num_frames = $trajectory->NumFrames;

if ($num_frames > 1) {
    print "Found $num_frames frames in the trajectory\n";

    # Open new xmol trajectory file once
    my $xmolFile = Documents->New("trj.txt");

    # Get atoms in the structure once
    my $atoms = $doc->UnitCell->Atoms;
    my $Natoms = @$atoms;

    # Pre-calculate conversion factor once
    my $conv_factor = 0.0433641042385997;

    # Loop over the frames
    my $start_frame = 1;  # Starting frame (modify as needed)
    my $end_frame = $num_frames;  # Ending frame (modify as needed)
    my $step = 1;  # Save every nth frame (modify as needed)

    # Loop over the frames with the specified range and step
    for (my $frame = $start_frame; $frame <= $end_frame; $frame += $step) {
        $trajectory->CurrentFrame = $frame;
        my $potentialEnergy = $doc->PotentialEnergy * $conv_factor; # Convert from kcal/mol to eV

        # Get lattice parameters and convert angles to radians for the current frame
        my $lattice = $doc->Lattice3D;
        my $a = $lattice->LengthA;
        my $b = $lattice->LengthB;
        my $c = $lattice->LengthC;
        
        my $pi =  3.141592653589793238462643383279;
        
        my $alpha = $lattice->AngleAlpha / 180 * $pi;
        my $beta = $lattice->AngleBeta / 180 * $pi;
        my $gamma = $lattice->AngleGamma / 180 * $pi;

        # Calculate lattice vectors for the current frame
        my $bc2 = $b**2 + $c**2 - 2 * $b * $c * cos($alpha);
        my $h1 = $a;
        my $h2 = $b * cos($gamma);
        my $h3 = $b * sin($gamma);
        my $h4 = $c * cos($beta);
        my $h5 = (($h2 - $h4)**2 + $h3**2 + $c**2 - $h4**2 - $bc2) / (2 * $h3);
        my $h6 = sqrt($c**2 - $h4**2 - $h5**2);

        # Write header xyz once per frame
        $xmolFile->Append("$Natoms\n");
        $xmolFile->Append(sprintf "energy=%.10f config_type=ms2xyz Lattice=\"%f %f %f %f %f %f %f %f %f\" pbc=\"T T T\" Properties=species:S:1:pos:R:3:force:R:3\n", $potentialEnergy, $h1, 0, 0, $h2, $h3, 0, $h4, $h5, $h6);

        # Loop over the atoms
        for my $atom (@$atoms) {
            # Write atom symbol, x-y-z- coordinates, and force vector
            my $force = $atom->Force;
            my $force_str = $force ? sprintf("%.10f %.10f %.10f", $force->X * $conv_factor, $force->Y * $conv_factor, $force->Z * $conv_factor) : "0 0 0";
            $xmolFile->Append(sprintf "%s %.10f %.10f %.10f %s\n", $atom->ElementSymbol, $atom->X, $atom->Y, $atom->Z, $force_str);
        }
    }

    # Close trajectory file once after all frames are processed
    $xmolFile->Close;
} else {
    print "The " . $doc->Name . " is not a multiframe trajectory file \n";
}