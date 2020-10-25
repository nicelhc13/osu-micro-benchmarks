#define BENCHMARK "OSU MPI%s All-to-All Personalized Exchange Latency Test"
/*
 * Copyright (C) 2002-2020 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include <osu_util_mpi.h>

int
main (int argc, char *argv[])
{
    int i, numprocs, rank, size;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double timer=0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0;
    char * sendbuf = NULL, * recvbuf = NULL;
    char * sendbuf_gpu = NULL, * recvbuf_gpu = NULL;
    char * tsendbuf_gpu = NULL, * trecvbuf_gpu = NULL;
    char * sendbuf_cpu = NULL, * recvbuf_cpu = NULL;
    int is_hh = 0, is_dd = 0;
    int po_ret;
    size_t bufsize;
    options.bench = COLLECTIVE;
    options.subtype = LAT;

    set_header(HEADER);
    set_benchmark_name("osu_alltoall");
    po_ret = process_options(argc, argv);

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));

    switch (po_ret) {
        case PO_BAD_USAGE:
            print_bad_usage_message(rank);
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        case PO_HELP_MESSAGE:
            print_help_message(rank);
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_VERSION_MESSAGE:
            print_version_message(rank);
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }

    if(numprocs < 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires at least two processes\n");
        }

        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    if ((options.max_message_size * numprocs) > options.max_mem_limit) {
        options.max_message_size = options.max_mem_limit / numprocs;
    }

    bufsize = options.max_message_size * numprocs;

    if (options.cpy_from_d) {
        if (cudaMalloc((void **) &recvbuf_gpu, bufsize)) {
            fprintf(stderr, "Error allocating receiving GPU memory %lu\n",
                    options.max_message_size);
        }
        if (cudaMalloc((void **) &sendbuf_gpu, bufsize)) {
            fprintf(stderr, "Error allocating sending GPU memory %lu\n",
                    options.max_message_size);
        }

        if (options.src == 'H' && options.dst == 'H') {
            if (rank == 0) {
                printf("** Host to Host\n");
            }
            is_hh = 1;
            if (options.add_serial) {
                printf("SDH HDS mode is enabled\n");
                if (cudaMalloc((void **) &trecvbuf_gpu, bufsize)) {
                    fprintf(stderr, "Error allocating "
                            "receiving tGPU memory %lu\n",
                            options.max_message_size);
                }
                if (cudaMalloc((void **) &tsendbuf_gpu, bufsize)) {
                    fprintf(stderr, "Error allocating "
                            "sending tGPU memory %lu\n",
                            options.max_message_size);
                }
            }
        } else if (options.src == 'D' && options.dst == 'D') {
            if (rank == 0) {
                printf("** Device to Device\n");
            }
            is_dd = 1;
        }
    } else if (options.cpy_from_c) {
        if ((recvbuf_cpu = (char *) malloc(bufsize)) == NULL) {
            fprintf(stderr, "Error allocating receiving CPU memory %lu\n",
                    options.max_message_size);
        }
        if ((sendbuf_cpu = (char *) malloc(bufsize)) == NULL) {
            fprintf(stderr, "Error allocating sending CPU memory %lu\n",
                    options.max_message_size);
        }

        if (options.src == 'H' && options.dst == 'H') {
            if (rank == 0) {
                printf("** Host to Host (Copy H to H)\n");
            }
            is_hh = 1;
        } else if (options.src == 'D' && options.dst == 'D') {
            if (rank == 0) {
                printf("** Device to Device (Copy H to D)\n");
            }
            is_dd = 1;
        }
    }

    if (allocate_memory_coll((void**)&sendbuf, bufsize, options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    }

    if (options.cpy_from_d && (is_hh || is_dd)) {
        enum accel_type old_accel = options.accel;
        options.accel = CUDA;
        set_buffer(sendbuf_gpu, options.accel, 1, bufsize);
        options.accel = old_accel;
    } else if (options.cpy_from_c && (is_hh || is_dd)) {
        enum accel_type old_accel = options.accel;
        options.accel = NONE;
        set_buffer(sendbuf_cpu, options.accel, 1, bufsize);
        options.accel = old_accel;
    } else {
        set_buffer(sendbuf, options.accel, 1, bufsize);
    }

    if (allocate_memory_coll((void**)&recvbuf, options.max_message_size * numprocs,
                options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    }

    set_buffer(recvbuf, options.accel, 0, bufsize);
    print_preamble(rank);

    for(size=options.min_message_size; size <= options.max_message_size; size *= 2) {
        if (size > LARGE_MESSAGE_SIZE) {
            options.skip = options.skip_large;
            options.iterations = options.iterations_large;
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        timer=0.0;

        for (i=0; i < options.iterations + options.skip ; i++) {
            t_start = MPI_Wtime();

            if (options.cpy_from_d) {
                if (is_hh) {
                    if (options.add_serial) {
                        cudaMemcpy(tsendbuf_gpu, sendbuf_gpu,
                                   bufsize, cudaMemcpyDeviceToDevice);
                        cudaMemcpy(sendbuf, tsendbuf_gpu,
                                   bufsize, cudaMemcpyDeviceToHost);
                    } else {
                        cudaMemcpy(sendbuf, sendbuf_gpu,
                                   bufsize, cudaMemcpyDeviceToHost);
                    }
                } else if (is_dd) {
                    cudaMemcpy(sendbuf, sendbuf_gpu,
                               bufsize, cudaMemcpyDeviceToDevice);
                }
            } else if (options.cpy_from_c) {
                if (is_hh) {
                    memcpy(sendbuf, sendbuf_cpu, bufsize);
                } else if (is_dd) {
                    cudaMemcpy(sendbuf, sendbuf_cpu,
                               bufsize, cudaMemcpyHostToDevice);
                }
            }
            MPI_CHECK(MPI_Alltoall(sendbuf, size, MPI_CHAR, recvbuf, size, MPI_CHAR,
                    MPI_COMM_WORLD));
            if (options.cpy_from_d) {
                if (is_hh) {
                    if (options.add_serial) {
                        cudaMemcpy(recvbuf_gpu, recvbuf,
                                   bufsize, cudaMemcpyHostToDevice);
                        cudaMemcpy(trecvbuf_gpu, recvbuf_gpu,
                                   bufsize, cudaMemcpyDeviceToDevice);
                    } else {
                        cudaMemcpy(recvbuf_gpu, recvbuf,
                                   bufsize, cudaMemcpyHostToDevice);
                    }
                } else if (is_dd) {
                    cudaMemcpy(recvbuf_gpu, recvbuf,
                               bufsize, cudaMemcpyDeviceToDevice);
                }
            } else if (options.cpy_from_c) {
                if (is_hh) {
                    memcpy(recvbuf_cpu, recvbuf, bufsize);
                } else if (is_dd) {
                    cudaMemcpy(recvbuf_cpu, recvbuf,
                               bufsize, cudaMemcpyDeviceToHost);
                }
            }


            t_stop = MPI_Wtime();

            if (i >= options.skip) {
                timer+=t_stop-t_start;
            }
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        }
        latency = (double)(timer * 1e6) / options.iterations;

        MPI_CHECK(MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0,
                MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD));
        avg_time = avg_time/numprocs;

        print_stats(rank, size, avg_time, min_time, max_time);
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    free_buffer(sendbuf, options.accel);
    free_buffer(recvbuf, options.accel);

    if (options.cpy_from_d) {
        cudaFree(sendbuf_gpu);
        cudaFree(recvbuf_gpu);
        if (options.add_serial) {
            cudaFree(tsendbuf_gpu);
            cudaFree(trecvbuf_gpu);
        }
    } else if (options.cpy_from_c) {
        free(sendbuf_cpu);
        free(recvbuf_cpu);
    }

    MPI_CHECK(MPI_Finalize());

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}

/* vi: set sw=4 sts=4 tw=80: */
