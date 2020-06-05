cd phonon\silicon_dispersion
del D.out omega2.out
..\..\..\src\phonon < input.txt
fc D.out D.out1
fc omega2.out omega2.out1
cd ..\..

