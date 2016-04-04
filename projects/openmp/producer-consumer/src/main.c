#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define PRODUCER 0


int
current_agent(int i) { return i % 2; }


void
parallel_producer_consumer(int *buffer, int size,
						   int *vec, int n,
						   int n_threads)
{
	int i, j;
	long long unsigned sum = 0;

	// Prevents successive threads creation by placing pragma omp
	// here instead of inner for statments.
	#pragma omp parallel num_threads(n_threads) \
						 private(i, j) \
	 					 reduction(+: sum)
		for(i = 0; i < n; i++)
			if(current_agent(i) == PRODUCER)
				#pragma omp for
				for (j = 0; j < size; j++)
					buffer[j] = vec[i] + j * vec[i + 1];
			else
				#pragma omp for
				for (j = 0; j < size; j++)
					sum += buffer[j];

	printf("%llu\n", sum);
}


void
sequential_producer_consumer(int *buffer, int size, int *vec, int n)
{
	int i, j;
	long long unsigned sum = 0;

	for(i = 0; i < n; i++)
		if(current_agent(i) == PRODUCER) for(j = 0; j < size; j++)
			buffer[j] = vec[i] + j * vec[i + 1];
		else for(j = 0; j < size; j++)
			sum += buffer[j];

	printf("%llu\n", sum);
}


int
main(int argc, char * argv[])
{
	double start, end;
	int *buff, *vec, i, n, size, thread_count;

	scanf("%d %d %d", &thread_count, &n, &size);

	buff = (int *)malloc(size * sizeof(int));
	vec = (int *)malloc(n * sizeof(int));

	for(i=0;i<n;i++)
		scanf("%d", &vec[i]);

	start = omp_get_wtime();
	parallel_producer_consumer(buff, size, vec, n, thread_count);
	// sequential_producer_consumer(buff, size, vec, n);
	end = omp_get_wtime();

	printf("%lf\n",end-start);

	free(buff);
	free(vec);

	return 0;
}


// $ cat /proc/cpuinfo
//
// processor	: 0
// vendor_id	: GenuineIntel
// cpu family	: 6
// model		: 60
// model name	: Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz
// stepping	: 3
// microcode	: 0x12
// cpu MHz		: 912.093
// cache size	: 6144 KB
// physical id	: 0
// siblings	: 8
// core id		: 0
// cpu cores	: 4
// apicid		: 0
// initial apicid	: 0
// fpu		: yes
// fpu_exception	: yes
// cpuid level	: 13
// wp		: yes
// flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 fma cx16 xtpr pdcm pcid sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm ida arat epb pln pts dtherm tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt
// bugs		:
// bogomips	: 4788.79
// clflush size	: 64
// cache_alignment	: 64
// address sizes	: 39 bits physical, 48 bits virtual
// power management:
//
// processor	: 1
// vendor_id	: GenuineIntel
// cpu family	: 6
// model		: 60
// model name	: Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz
// stepping	: 3
// microcode	: 0x12
// cpu MHz		: 873.750
// cache size	: 6144 KB
// physical id	: 0
// siblings	: 8
// core id		: 0
// cpu cores	: 4
// apicid		: 1
// initial apicid	: 1
// fpu		: yes
// fpu_exception	: yes
// cpuid level	: 13
// wp		: yes
// flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 fma cx16 xtpr pdcm pcid sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm ida arat epb pln pts dtherm tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt
// bugs		:
// bogomips	: 4788.79
// clflush size	: 64
// cache_alignment	: 64
// address sizes	: 39 bits physical, 48 bits virtual
// power management:
//
// processor	: 2
// vendor_id	: GenuineIntel
// cpu family	: 6
// model		: 60
// model name	: Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz
// stepping	: 3
// microcode	: 0x12
// cpu MHz		: 999.562
// cache size	: 6144 KB
// physical id	: 0
// siblings	: 8
// core id		: 1
// cpu cores	: 4
// apicid		: 2
// initial apicid	: 2
// fpu		: yes
// fpu_exception	: yes
// cpuid level	: 13
// wp		: yes
// flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 fma cx16 xtpr pdcm pcid sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm ida arat epb pln pts dtherm tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt
// bugs		:
// bogomips	: 4788.79
// clflush size	: 64
// cache_alignment	: 64
// address sizes	: 39 bits physical, 48 bits virtual
// power management:
//
// processor	: 3
// vendor_id	: GenuineIntel
// cpu family	: 6
// model		: 60
// model name	: Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz
// stepping	: 3
// microcode	: 0x12
// cpu MHz		: 1124.906
// cache size	: 6144 KB
// physical id	: 0
// siblings	: 8
// core id		: 1
// cpu cores	: 4
// apicid		: 3
// initial apicid	: 3
// fpu		: yes
// fpu_exception	: yes
// cpuid level	: 13
// wp		: yes
// flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 fma cx16 xtpr pdcm pcid sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm ida arat epb pln pts dtherm tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt
// bugs		:
// bogomips	: 4788.79
// clflush size	: 64
// cache_alignment	: 64
// address sizes	: 39 bits physical, 48 bits virtual
// power management:
//
// processor	: 4
// vendor_id	: GenuineIntel
// cpu family	: 6
// model		: 60
// model name	: Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz
// stepping	: 3
// microcode	: 0x12
// cpu MHz		: 843.937
// cache size	: 6144 KB
// physical id	: 0
// siblings	: 8
// core id		: 2
// cpu cores	: 4
// apicid		: 4
// initial apicid	: 4
// fpu		: yes
// fpu_exception	: yes
// cpuid level	: 13
// wp		: yes
// flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 fma cx16 xtpr pdcm pcid sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm ida arat epb pln pts dtherm tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt
// bugs		:
// bogomips	: 4788.79
// clflush size	: 64
// cache_alignment	: 64
// address sizes	: 39 bits physical, 48 bits virtual
// power management:
//
// processor	: 5
// vendor_id	: GenuineIntel
// cpu family	: 6
// model		: 60
// model name	: Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz
// stepping	: 3
// microcode	: 0x12
// cpu MHz		: 897.281
// cache size	: 6144 KB
// physical id	: 0
// siblings	: 8
// core id		: 2
// cpu cores	: 4
// apicid		: 5
// initial apicid	: 5
// fpu		: yes
// fpu_exception	: yes
// cpuid level	: 13
// wp		: yes
// flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 fma cx16 xtpr pdcm pcid sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm ida arat epb pln pts dtherm tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt
// bugs		:
// bogomips	: 4788.79
// clflush size	: 64
// cache_alignment	: 64
// address sizes	: 39 bits physical, 48 bits virtual
// power management:
//
// processor	: 6
// vendor_id	: GenuineIntel
// cpu family	: 6
// model		: 60
// model name	: Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz
// stepping	: 3
// microcode	: 0x12
// cpu MHz		: 932.343
// cache size	: 6144 KB
// physical id	: 0
// siblings	: 8
// core id		: 3
// cpu cores	: 4
// apicid		: 6
// initial apicid	: 6
// fpu		: yes
// fpu_exception	: yes
// cpuid level	: 13
// wp		: yes
// flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 fma cx16 xtpr pdcm pcid sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm ida arat epb pln pts dtherm tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt
// bugs		:
// bogomips	: 4788.79
// clflush size	: 64
// cache_alignment	: 64
// address sizes	: 39 bits physical, 48 bits virtual
// power management:
//
// processor	: 7
// vendor_id	: GenuineIntel
// cpu family	: 6
// model		: 60
// model name	: Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz
// stepping	: 3
// microcode	: 0x12
// cpu MHz		: 1024.875
// cache size	: 6144 KB
// physical id	: 0
// siblings	: 8
// core id		: 3
// cpu cores	: 4
// apicid		: 7
// initial apicid	: 7
// fpu		: yes
// fpu_exception	: yes
// cpuid level	: 13
// wp		: yes
// flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 fma cx16 xtpr pdcm pcid sse4_1 sse4_2 movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm ida arat epb pln pts dtherm tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt
// bugs		:
// bogomips	: 4788.79
// clflush size	: 64
// cache_alignment	: 64
// address sizes	: 39 bits physical, 48 bits virtual
// power management:
//
//

// Profile results
// ---------------
//
// arq1.in
//
// %   cumulative   self              self     total
// time   seconds   seconds    calls  Ts/call  Ts/call  name
// 0.00      0.00     0.00     1000     0.00     0.00  current_agent
// 0.00      0.00     0.00        1     0.00     0.00  sequential_producer_consumer
//
// %   cumulative   self              self     total
// time   seconds   seconds    calls  Ts/call  Ts/call  name
// 0.00      0.00     0.00     1331     0.00     0.00  current_agent
// 0.00      0.00     0.00        1     0.00     0.00  main
// 0.00      0.00     0.00        1     0.00     0.00  parallel_producer_consumer
//
// arq2.in
//
// Each sample counts as 0.01 seconds.
//   %   cumulative   self              self     total
//  time   seconds   seconds    calls  ms/call  ms/call  name
// 100.50      0.31     0.31        1   311.54   311.54  sequential_producer_consumer
//   0.00      0.31     0.00    10000     0.00     0.00  current_agent
//
// Each sample counts as 0.01 seconds.
//   %   cumulative   self              self     total
// time   seconds   seconds    calls  ms/call  ms/call  name
// 98.63      0.27     0.27        1   266.31   266.31  main
//  0.00      0.27     0.00    15049     0.00     0.00  current_agent
//  0.00      0.27     0.00        1     0.00     0.00  parallel_producer_consumer
//
// arq3.in
//
// Each sample counts as 0.01 seconds.
//    %   cumulative   self              self     total
//   time   seconds   seconds    calls   s/call   s/call  name
//  100.11      2.62     2.62        1     2.62     2.63  sequential_producer_consumer
//    0.38      2.63     0.01   100000     0.00     0.00  current_agent
//
// Each sample counts as 0.01 seconds.
//  %   cumulative   self              self     total
// time   seconds   seconds    calls   s/call   s/call  name
// 98.50      2.73     2.73        1     2.73     2.73  main
//  0.00      2.73     0.00   137780     0.00     0.00  current_agent
//  0.00      2.73     0.00        1     0.00     0.00  parallel_producer_consumer
//
//

// Compilation optimization results
// --------------------------------
//
// Notes: -O0 option wasn't used, as [1] states that it's the default
// compilation option and therefore has no difference from omitting the
// parameter altogether.
//
// References
// ----------
// [1] https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
//
// $ gcc -std=c99 -pedantic -Wall -fopenmp -lm c.c -o c_sequential
// $ ./c_sequential < arq1.in
// 12784254000
// 0.004386
// $ ./c_sequential < arq2.in
// 12426110415000
// 0.270412
// $ ./c_sequential < arq3.in
// 126047050705000
// 2.601086
// $ gcc -std=c99 -pedantic -Wall -fopenmp -lm -O1 c.c -o c_sequential
// $ ./c_sequential < arq1.in
// 12784254000
// 0.000696
// $ ./c_sequential < arq2.in
// 12426110415000
// 0.048660
// $ ./c_sequential < arq3.in
// 126047050705000
// 0.527403
// $ gcc -std=c99 -pedantic -Wall -fopenmp -lm -O2 c.c -o c_sequential
// $ ./c_sequential < arq1.in
// 12784254000
// 0.001637
// $ ./c_sequential < arq2.in
// 12426110415000
// 0.098308
// $ ./c_sequential < arq3.in
// 126047050705000
// 0.592873
// $ gcc -std=c99 -pedantic -Wall -fopenmp -lm -O3 c.c -o c_sequential
// $ ./c_sequential < arq1.in
// 12784254000
// 0.000941
// $ ./c_sequential < arq2.in
// 12426110415000
// 0.040222
// $ ./c_sequential < arq3.in
// 126047050705000
// 0.339946
