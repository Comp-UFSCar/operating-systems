# Zip Password Cracking

Parallel program for cracking zip files' passwords, using `pthreads` library.

## Reports

```shell
# Serial password cracking time results.
ldavid@jane:...$ ./pass_serial < arq1.in
Senha:10000
16.141632
ldavid@jane:...$ ./pass_serial < arq2.in
Senha:100000
160.687960
ldavid@jane:...$ ./pass_serial < arq3.in
Senha:450000
764.640047
ldavid@jane:...$ ./pass_serial < arq4.in
Senha:310000
538.292841

# Parallel password cracking time results.
ldavid@jane:...$$ ./pass_parallel < arq1.in
Senha:10000
7.850181
ldavid@jane:...$$ ./pass_parallel < arq2.in
Senha:100000
80.933334
ldavid@jane:...$$ ./pass_parallel < arq3.in
Senha:450000
350.083878
ldavid@jane:...$$ ./pass_parallel < arq4.in
Senha:310000
247.497154

```
