./rendezvous-parallel 1 4
cp parallel-out.txt parallel-out14.txt
./rendezvous-parallel 4 4
cp parallel-out.txt parallel-out44.txt
./rendezvous-parallel 4 3
cp parallel-out.txt parallel-out43.txt
./rendezvous-parallel 4 2
cp parallel-out.txt parallel-out42.txt
./rendezvous-serial 3
cp gmon.out gmon3.out
cp serial-out.txt serial-out3.txt
./rendezvous-serial 4
cp serial-out.txt serial-out4.txt
cp gmon.out gmon4.out