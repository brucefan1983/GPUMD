import jse.atom.data.DataXYZ
import jse.code.IO
import jse.math.matrix.IMatrix
import jsex.nnap.NNAP

def pot = new NNAP('../../../potentials/nnap/Cu.json')
def data = DataXYZ.read('dump.1.xyz')

double eng = (double)data.parameter('energy')
def force = (IMatrix)data.property('forces')
def stress = IO.Text.str2data((String)data.parameter('virial'))
stress.div2this(data.volume())
stress.negative2this()

double engStd = pot.calEnergy(data)
def forceStd = pot.calForces(data)
def (sxx, syy, szz, sxy, sxz, syz) = pot.calStress(data)

println("energy diff: ${engStd-eng}")
println("force diff: ${(forceStd-force).asVecRow().abs().mean()}")
println("stress diff: [${sxx-stress[0]}, ${syy-stress[4]}, ${szz-stress[8]}, ${sxy-stress[1]}, ${sxz-stress[2]}, ${syz-stress[5]}]")


