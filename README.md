# MD5 Cracker

This Code is used to bruteforce for md5 hash. md5_gpu.cu can run on multiple Nvidia gpus at the same time. md5_multi.cu can run on multi-node multi-gpu setup.

## Requirements

* GNU Make
* Nvidia CUDA compiler
* OpenMPI

## Compilation

To compile the code we need to install openMPI and CUDA, make sure to setup correct `arch` in Makefile and then:

```bash
$ make gpu
```

The compilation step creates an executable file called `md5_gpu`.

```bash
$ make multi
```

The compilation step creates an executable file called `md5_multi`.

## Usage

The program detects and uses all available CUDA-enabled GPUs within a single node. To run it:

```bash
$ ./md5_gpu --charset=CHARSET --hash=HASH

Info: # device(s) found
The password for MD5 hash HASH is PASSWORD
```

The program detects and uses all available CUDA-enabled GPUs on all nodes node. To run it:

```bash
$ mpirun --pernode --machinefile ./nodefile ./md5_multi --charset=CHARSET --hash=HASH --min_len 5 

Info: # device(s) found
The password for MD5 hash HASH is PASSWORD
```

where:

* `CHARSET` is either `0` for numeric passwords or `1` for alphanumeric.
* `HASH` is an MD5 hash as 32 hexadecimal digits.
* `min_len` is the minimum length of which the program starts to bruteforce.
* `max_len` is the maximum length upto which the program tries to bruteforce.
* `./nodefile` is the file specifing the the node names.

## Attribution

Reference:  [iryont/md5-cracker](https://github.com/iryont/md5-cracker).

Reference: [mkappas/challenge_2_participant_info](https://github.com/mkappas/challenge_2_participant_info)
