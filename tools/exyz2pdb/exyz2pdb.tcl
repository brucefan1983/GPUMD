proc read_exyz {filename} {
    set file [open $filename "r"]
    set structures {}

    while {[gets $file line] >= 0} {
        set natoms [string trim $line]
        if {![gets $file line]} break

        set lattice {1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0}
        
        if {[regexp {Lattice="([^"]+)"} $line -> lattice_values]} {
            set lattice [split $lattice_values]
        }

        # Extract lattice vectors
        set a1 [lindex $lattice 0]; set a2 [lindex $lattice 1]; set a3 [lindex $lattice 2]
        set b1 [lindex $lattice 3]; set b2 [lindex $lattice 4]; set b3 [lindex $lattice 5]
        set c1 [lindex $lattice 6]; set c2 [lindex $lattice 7]; set c3 [lindex $lattice 8]

        # Calculate cell lengths
        set a [expr {sqrt($a1*$a1 + $a2*$a2 + $a3*$a3)}]
        set b [expr {sqrt($b1*$b1 + $b2*$b2 + $b3*$b3)}]
        set c [expr {sqrt($c1*$c1 + $c2*$c2 + $c3*$c3)}]

        # Calculate angles
        set alpha [expr {acos(($b1*$c1 + $b2*$c2 + $b3*$c3) / ($b*$c)) * 180 / 3.1415926535}]
        set beta [expr {acos(($a1*$c1 + $a2*$c2 + $a3*$c3) / ($a*$c)) * 180 / 3.1415926535}]
        set gamma [expr {acos(($a1*$b1 + $a2*$b2 + $a3*$b3) / ($a*$b)) * 180 / 3.1415926535}]

        set atoms {}
        for {set i 0} {$i < $natoms} {incr i} {
            gets $file line
            lappend atoms $line
        }

        lappend structures [list $a $b $c $alpha $beta $gamma $atoms]
    }
    close $file
    return $structures
}

proc write_pdb {structures output_filename} {
    set pdb_file [open $output_filename "w"]

    foreach structure $structures {
        lassign $structure a b c alpha beta gamma atoms

        puts $pdb_file [format "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1" $a $b $c $alpha $beta $gamma]

        set atom_index 1
        foreach atom $atoms {
            lassign [split $atom] element x y z
            puts $pdb_file [format "ATOM  %5d %-4s MOL     1    %8.3f%8.3f%8.3f  1.00  0.00          %-2s" $atom_index $element $x $y $z $element]
            incr atom_index
        }
        
        puts $pdb_file "END"
    }
    close $pdb_file
}

# 使用示例
set structures [read_exyz "2.xyz"]
set pdb_filename "output_file.pdb"
write_pdb $structures $pdb_filename

# 加载生成的 PDB 文件并在 VMD 中显示
mol new $pdb_filename